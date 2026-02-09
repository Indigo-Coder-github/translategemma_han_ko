"""조선왕조실록 국역(한국어 번역) 수집 스크립트.

sillok.history.go.kr의 JSON API를 사용하여 국역 텍스트를 수집하고,
parse_sillok.py로 추출한 원문 데이터와 매칭하여 저장한다.

일자(日) 단위로 배치 요청하여 서버 부하를 줄인다.
(기사 단위 41만 회 → 일자 단위 16만 회로 요청 62% 절감)

JSON API endpoint:
    GET https://sillok.history.go.kr/search/collectView.do?id={day_id}

    일자 ID (예: waa_10107017)로 요청하면 그 날의 모든 기사를 반환.
    응답의 sillokResult[]에 국역(k 접두사)과 원문(w 접두사)이 함께 포함.

Usage:
    python scripts/scrape_sillok_korean.py
    python scripts/scrape_sillok_korean.py --delay-min 1 --delay-max 5
    python scripts/scrape_sillok_korean.py --king 태조
    python scripts/scrape_sillok_korean.py --resume
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE_URL = "https://sillok.history.go.kr/search/collectView.do"
PROGRESS_FILE_SUFFIX = ".progress"


def article_id_to_day_id(article_id: str) -> str:
    """기사 ID에서 일자 ID를 추출한다.

    예: waa_10107017_001 → waa_10107017
    """
    parts = article_id.split("_")
    return parts[0] + "_" + parts[1]


def article_id_to_korean_id(article_id: str) -> str:
    """원문 기사 ID를 국역 ID로 변환한다.

    예: waa_10107017_001 → kaa_10107017_001
    """
    return "k" + article_id[1:]


def strip_html_tags(text: str) -> str:
    """HTML 태그를 제거하고 텍스트만 추출한다."""
    text = re.sub(r"<[^>]+>", "", text)
    text = _decode_html_entities(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_translation_from_html(html: str) -> str:
    """contentHg HTML에서 번역 텍스트를 추출한다.

    content 필드는 각주(역자 주석)가 본문에 인라인되어 섞이지만,
    contentHg는 각주를 <a class="footnote_super"><sup>001)</sup></a>
    마커로만 표시하고 각주 본문은 footnoteHg에 분리되어 있다.

    처리:
    - <sup> 태그 제거 (각주 번호 001), 002) 등)
    - 나머지 HTML 태그 제거 (텍스트 보존)
    - HTML 엔티티 디코딩, 다중 공백 정리
    """
    # <sup>...</sup> 제거 (각주 번호)
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", html)
    # 나머지 HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", text)
    text = _decode_html_entities(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_footnotes(footnote_html: str) -> str:
    """footnoteHg HTML에서 각주 텍스트를 추출한다.

    입력 예:
        <li><a href="#footnote_view1" id="footnote_1">[註 001]</a>
            시좌궁(時坐宮) : 그 당시에 왕이 거처하던 궁전.</li>

    출력: "[註 001] 시좌궁(時坐宮) : 그 당시에 왕이 거처하던 궁전."
    """
    text = re.sub(r"<[^>]+>", "", footnote_html)
    text = _decode_html_entities(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _decode_html_entities(text: str) -> str:
    """HTML 엔티티를 디코딩한다."""
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    return text


def fetch_day_translations(
    day_id: str,
    session: requests.Session,
    timeout: int = 30,
) -> dict[str, dict[str, str | None]]:
    """일자 ID로 해당 일자의 모든 국역을 가져온다.

    Returns:
        {
            article_id: {
                "translation": 국역 텍스트 또는 None,
                "korean_id": 국역 ID 또는 None,
            },
            ...
        }
        또는 에러 시 빈 dict + "_error" 키에 에러 메시지
    """
    result: dict[str, dict[str, str | None]] = {}

    try:
        resp = session.get(
            BASE_URL,
            params={"id": day_id},
            timeout=timeout,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "application/json" not in content_type:
            result["_error"] = {"translation": None, "korean_id": None, "error": "not_json_response"}
            return result

        data = resp.json()
    except requests.RequestException as e:
        result["_error"] = {"translation": None, "korean_id": None, "error": f"request_error: {e}"}
        return result
    except json.JSONDecodeError as e:
        result["_error"] = {"translation": None, "korean_id": None, "error": f"json_error: {e}"}
        return result

    sillok_results = data.get("sillokResult") or []

    for sr in sillok_results:
        sr_id = sr.get("id", "")
        # 국역 항목만 (k 접두사)
        if not sr_id.startswith("k"):
            continue

        # k → w로 변환하여 원문 기사 ID 복원
        original_id = "w" + sr_id[1:]

        content = sr.get("content")
        content_hg = sr.get("contentHg")
        footnote_hg = sr.get("footnoteHg")

        # contentHg 우선 사용 (content에는 각주가 본문에 인라인됨)
        translation = None
        if content_hg and isinstance(content_hg, str):
            translation = extract_translation_from_html(content_hg)
        elif content and isinstance(content, str):
            translation = strip_html_tags(content)

        footnotes = None
        if footnote_hg and isinstance(footnote_hg, str):
            footnotes = extract_footnotes(footnote_hg)

        result[original_id] = {
            "translation": translation,
            "korean_id": sr_id,
            "footnotes": footnotes,
        }

    return result


def load_progress(progress_path: Path) -> set[str]:
    """완료된 일자 ID들을 로드한다."""
    if not progress_path.exists():
        return set()
    with open(progress_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_progress(progress_path: Path, day_id: str) -> None:
    """완료된 일자 ID를 기록한다."""
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(day_id + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="조선왕조실록 국역 수집 (일자 배치)")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/parsed/sillok/articles.jsonl"),
        help="파싱된 기사 JSONL (default: data/parsed/sillok/articles.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/parsed/sillok/articles_with_korean.jsonl"),
        help="출력 JSONL (default: data/parsed/sillok/articles_with_korean.jsonl)",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=1.0,
        help="최소 대기 시간(초) (default: 1.0)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=5.0,
        help="최대 대기 시간(초) (default: 5.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP 요청 타임아웃(초) (default: 30)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="실패 시 재시도 횟수 (default: 3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="중단된 지점부터 재개",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="수집할 최대 일자 수 (0=전체, 테스트용)",
    )
    parser.add_argument(
        "--king",
        type=str,
        default="",
        help="특정 왕대만 수집 (예: 태조, 세종)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    progress_path = output_path.with_suffix(output_path.suffix + PROGRESS_FILE_SUFFIX)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 입력 기사를 일자별로 그룹핑
    # day_id → [article, article, ...]
    day_groups: dict[str, list[dict]] = defaultdict(list)
    skipped_non_api = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            if args.king and article.get("king") != args.king:
                continue
            aid = article["article_id"]
            parts = aid.split("_")
            if len(parts) != 3:
                skipped_non_api += 1
                continue
            day_id = article_id_to_day_id(aid)
            day_groups[day_id].append(article)

    total_articles = sum(len(v) for v in day_groups.values())
    total_days = len(day_groups)
    print(f"[INFO] 대상: 기사 {total_articles:,}건, 일자 {total_days:,}일")
    if skipped_non_api:
        print(f"[INFO] API 미지원 (총서/부록): {skipped_non_api:,}건 제외")

    # 진행 상황 확인
    completed_days: set[str] = set()
    if args.resume:
        completed_days = load_progress(progress_path)
        print(f"[INFO] 이전 진행: {len(completed_days):,}일 완료")
    else:
        if output_path.exists():
            output_path.unlink()
        if progress_path.exists():
            progress_path.unlink()

    # 남은 일자들
    day_ids_sorted = sorted(
        [d for d in day_groups if d not in completed_days]
    )
    if args.limit > 0:
        day_ids_sorted = day_ids_sorted[: args.limit]

    remaining_articles = sum(len(day_groups[d]) for d in day_ids_sorted)
    avg_delay = (args.delay_min + args.delay_max) / 2
    eta_h = len(day_ids_sorted) * avg_delay / 3600

    print(f"[INFO] 수집 대상: {len(day_ids_sorted):,}일 ({remaining_articles:,}건)")
    if day_ids_sorted:
        first_day = day_ids_sorted[0]
        first_king = day_groups[first_day][0]["king"] if day_groups[first_day] else "?"
        last_day = day_ids_sorted[-1]
        last_king = day_groups[last_day][0]["king"] if day_groups[last_day] else "?"
        print(f"[INFO] 시작: {first_king} {first_day} → 끝: {last_king} {last_day}")
    print(f"[INFO] 대기: {args.delay_min}~{args.delay_max}초 (평균 {avg_delay:.1f}초)")
    print(f"[INFO] 예상 소요: {eta_h:.1f}시간 ({eta_h/24:.1f}일)")
    print()

    # 세션 설정
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://sillok.history.go.kr/",
    })

    # 통계
    stats = {
        "success": 0,
        "no_translation": 0,
        "error": 0,
        "days_done": 0,
        "requests": 0,
    }
    start_time = time.time()

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, day_id in enumerate(day_ids_sorted, 1):
            articles_in_day = day_groups[day_id]

            # API 요청 (재시도 포함)
            day_result = None
            for attempt in range(1, args.max_retries + 1):
                day_result = fetch_day_translations(
                    day_id, session, args.timeout
                )
                stats["requests"] += 1

                if "_error" in day_result:
                    err = day_result["_error"].get("error", "")
                    if "request_error" in str(err) and attempt < args.max_retries:
                        wait = avg_delay * (2 ** attempt)
                        print(
                            f"  [RETRY] {day_id} attempt {attempt}/{args.max_retries}, "
                            f"waiting {wait:.1f}s..."
                        )
                        time.sleep(wait)
                        continue
                break

            has_error = "_error" in day_result

            # 일자 내 각 기사에 국역 매칭
            for article in articles_in_day:
                aid = article["article_id"]
                article_out = {**article}

                if has_error:
                    article_out["translation"] = None
                    article_out["korean_id"] = None
                    article_out["footnotes"] = None
                    stats["error"] += 1
                elif aid in day_result:
                    kr = day_result[aid]
                    article_out["translation"] = kr["translation"]
                    article_out["korean_id"] = kr["korean_id"]
                    article_out["footnotes"] = kr.get("footnotes")
                    if kr["translation"]:
                        stats["success"] += 1
                    else:
                        stats["no_translation"] += 1
                else:
                    article_out["translation"] = None
                    article_out["korean_id"] = None
                    article_out["footnotes"] = None
                    stats["no_translation"] += 1

                out_f.write(json.dumps(article_out, ensure_ascii=False) + "\n")

            save_progress(progress_path, day_id)
            stats["days_done"] += 1

            # 진행 표시
            if i % 500 == 0 or i == len(day_ids_sorted):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta_sec = (len(day_ids_sorted) - i) / rate if rate > 0 else 0
                eta_h_now = eta_sec / 3600

                king = articles_in_day[0]["king"] if articles_in_day else "?"
                print(
                    f"  [{i:,}/{len(day_ids_sorted):,}일] "
                    f"{king} {day_id} | "
                    f"성공={stats['success']:,} "
                    f"국역없음={stats['no_translation']:,} "
                    f"오류={stats['error']:,} | "
                    f"요청={stats['requests']:,} | "
                    f"ETA {eta_h_now:.1f}h"
                )
                out_f.flush()

            # 랜덤 대기
            delay = random.uniform(args.delay_min, args.delay_max)
            time.sleep(delay)

    elapsed_total = time.time() - start_time
    print(f"\n[DONE] {elapsed_total / 3600:.1f}시간 소요")
    print(f"  API 요청: {stats['requests']:,}회")
    print(f"  성공: {stats['success']:,}건")
    print(f"  국역 없음: {stats['no_translation']:,}건")
    print(f"  오류: {stats['error']:,}건")
    print(f"  출력: {output_path}")
    print(f"\n  --resume 옵션으로 중단 시 재개 가능")


if __name__ == "__main__":
    main()
