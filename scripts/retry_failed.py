"""국역 수집 실패 일자를 재시도하는 스크립트.

articles_with_korean.jsonl에서 API 오류로 국역이 누락된 일자를 찾아
재시도하고 결과를 업데이트한다.

에러 일자 판별 기준 (기본):
    해당 일자의 모든 기사가 translation=null → API 에러로 간주.
    (API 성공 시에는 국역 없는 기사만 개별적으로 null이 되므로,
     전체 null은 요청 자체가 실패했을 가능성이 높다.)

--all-null 옵션:
    translation=null인 기사가 하나라도 있는 일자 전부 재시도.
    간헐적 API 불안정으로 일부만 누락된 경우에 유용.

Usage:
    python scripts/retry_failed.py                     # 에러 일자만 재시도
    python scripts/retry_failed.py --all-null           # null 포함 일자 전부
    python scripts/retry_failed.py --dry-run            # 대상만 확인
    python scripts/retry_failed.py --king 태조          # 특정 왕대만
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# 같은 디렉토리의 scrape_sillok_korean에서 함수 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_sillok_korean import (
    article_id_to_day_id,
    fetch_day_translations,
)


def identify_error_days(
    articles: list[dict],
    all_null: bool = False,
) -> tuple[set[str], dict[str, int]]:
    """에러로 추정되는 일자 ID들을 찾는다.

    기본: 일자 내 모든 기사가 translation=null이면 에러 일자.
    all_null: translation=null인 기사가 하나라도 있는 일자 전부.

    Returns:
        (error_days, day_null_counts): 에러 일자 set, 일자별 null 기사 수
    """
    day_groups: dict[str, list[dict]] = defaultdict(list)

    for article in articles:
        aid = article.get("article_id", "")
        parts = aid.split("_")
        if len(parts) != 3:
            continue
        day_id = article_id_to_day_id(aid)
        day_groups[day_id].append(article)

    error_days: set[str] = set()
    day_null_counts: dict[str, int] = {}

    for day_id, day_articles in day_groups.items():
        null_count = sum(1 for a in day_articles if not a.get("translation"))
        if null_count == 0:
            continue
        day_null_counts[day_id] = null_count

        if all_null:
            error_days.add(day_id)
        else:
            # 전부 null이면 API 에러로 간주
            if null_count == len(day_articles):
                error_days.add(day_id)

    return error_days, day_null_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="국역 수집 실패 일자 재시도",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/parsed/sillok/articles_with_korean.jsonl"),
        help="수집된 기사 JSONL (default: data/parsed/sillok/articles_with_korean.jsonl)",
    )
    parser.add_argument(
        "--all-null",
        action="store_true",
        help="translation=null인 기사가 하나라도 있는 일자 전부 재시도 "
        "(기본: 일자 내 전체 null인 경우만)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="재시도 대상만 출력하고 실제 요청은 하지 않음",
    )
    parser.add_argument(
        "--king",
        type=str,
        default="",
        help="특정 왕대만 재시도 (예: 태조, 세종)",
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
        "--no-backup",
        action="store_true",
        help="백업 파일을 생성하지 않음",
    )
    args = parser.parse_args()

    input_path: Path = args.input

    if not input_path.exists():
        print(f"[ERROR] 파일 없음: {input_path}")
        sys.exit(1)

    # ── 1. 전체 기사 로드 ──
    print(f"[INFO] 로딩: {input_path}")
    articles: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    print(f"[INFO] 총 {len(articles):,}건 로드")

    # 왕대 필터
    if args.king:
        target_articles = [a for a in articles if a.get("king") == args.king]
        print(f"[INFO] {args.king} 필터: {len(target_articles):,}건")
    else:
        target_articles = articles

    # ── 2. 에러 일자 식별 ──
    error_days, day_null_counts = identify_error_days(target_articles, args.all_null)

    error_article_count = 0
    for a in target_articles:
        aid = a.get("article_id", "")
        if len(aid.split("_")) == 3 and article_id_to_day_id(aid) in error_days:
            error_article_count += 1

    null_article_total = sum(day_null_counts.values())
    mode = "--all-null (null 포함 일자 전부)" if args.all_null else "전체-null 일자만 (API 에러 추정)"
    print(f"[INFO] 모드: {mode}")
    print(f"[INFO] null 기사 총: {null_article_total:,}건 ({len(day_null_counts):,}일)")
    print(f"[INFO] 재시도 대상: {len(error_days):,}일 ({error_article_count:,}건)")

    if not error_days:
        print("[INFO] 재시도할 일자가 없습니다.")
        return

    # ── 3. dry-run ──
    if args.dry_run:
        print(f"\n[DRY-RUN] 재시도 대상 일자 (상위 30개):")
        for day_id in sorted(error_days)[:30]:
            print(f"  {day_id}  (null {day_null_counts.get(day_id, '?')}건)")
        if len(error_days) > 30:
            print(f"  ... 외 {len(error_days) - 30}일")
        return

    # ── 4. API 재시도 ──
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://sillok.history.go.kr/",
        }
    )

    day_results: dict[str, dict] = {}
    avg_delay = (args.delay_min + args.delay_max) / 2
    sorted_days = sorted(error_days)
    stats = {
        "api_requests": 0,
        "days_success": 0,
        "days_still_error": 0,
        "recovered": 0,
        "still_null": 0,
    }

    eta_h = len(sorted_days) * avg_delay / 3600
    print(f"\n[INFO] {len(sorted_days):,}일 재시도 시작 (예상 {eta_h:.1f}시간)...")
    start_time = time.time()

    for i, day_id in enumerate(sorted_days, 1):
        result = None
        for attempt in range(1, args.max_retries + 1):
            result = fetch_day_translations(day_id, session, args.timeout)
            stats["api_requests"] += 1

            if "_error" in result:
                if attempt < args.max_retries:
                    wait = avg_delay * (2**attempt)
                    print(
                        f"  [RETRY] {day_id} attempt {attempt}/{args.max_retries}, "
                        f"waiting {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue
            break

        day_results[day_id] = result

        if "_error" in result:
            stats["days_still_error"] += 1
        else:
            stats["days_success"] += 1

        if i % 100 == 0 or i == len(sorted_days):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta_sec = (len(sorted_days) - i) / rate if rate > 0 else 0
            print(
                f"  [{i:,}/{len(sorted_days):,}일] "
                f"성공={stats['days_success']} "
                f"실패={stats['days_still_error']} | "
                f"ETA {eta_sec / 60:.1f}min"
            )

        delay = random.uniform(args.delay_min, args.delay_max)
        time.sleep(delay)

    # ── 5. 기사 업데이트 ──
    print("\n[INFO] 기사 업데이트 중...")
    updated_articles: list[dict] = []

    for article in articles:
        aid = article.get("article_id", "")
        parts = aid.split("_")

        if len(parts) != 3:
            updated_articles.append(article)
            continue

        day_id = article_id_to_day_id(aid)

        if day_id not in day_results:
            updated_articles.append(article)
            continue

        result = day_results[day_id]

        if "_error" in result:
            # 여전히 에러 → 기존 값 유지
            updated_articles.append(article)
            continue

        if aid in result:
            kr = result[aid]
            new_article = {**article}
            new_article["translation"] = kr["translation"]
            new_article["korean_id"] = kr["korean_id"]
            new_article["footnotes"] = kr.get("footnotes")

            if kr["translation"]:
                stats["recovered"] += 1
            else:
                stats["still_null"] += 1
            updated_articles.append(new_article)
        else:
            # API 성공했지만 이 기사의 국역이 없음 → 원래 없는 것
            stats["still_null"] += 1
            updated_articles.append(article)

    # ── 6. 파일 쓰기 ──
    if not args.no_backup:
        backup_path = input_path.with_suffix(".jsonl.bak")
        print(f"[INFO] 백업: {backup_path}")
        import shutil

        shutil.copy2(input_path, backup_path)

    with open(input_path, "w", encoding="utf-8") as f:
        for article in updated_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    elapsed_total = time.time() - start_time
    print(f"\n[DONE] {elapsed_total / 60:.1f}분 소요")
    print(f"  API 요청: {stats['api_requests']:,}회")
    print(f"  일자 성공: {stats['days_success']:,} / 실패: {stats['days_still_error']:,}")
    print(f"  복구 성공: {stats['recovered']:,}건")
    print(f"  여전히 null: {stats['still_null']:,}건")
    print(f"  출력: {input_path}")


if __name__ == "__main__":
    main()
