"""데이터 정합성 검증 스크립트.

수집/전처리 데이터의 품질 문제를 탐지한다:
- 한자 불일치: 번역의 (漢字)가 원문에 미포함
- 중복 용어: 동일 term(漢字)가 번역에 2회+ 등장 (각주 인라인 삽입 의심)
- 인라인 각주: (漢字) 짧은설명. 패턴 잔존
- trailing space: (漢字) 조사 패턴 (공백이 남은 경우)

Usage:
    python scripts/validate_pairs.py --input data/parsed/sillok/articles_with_korean.jsonl
    python scripts/validate_pairs.py --input data/processed/sillok/clean_pairs.jsonl --processed
    python scripts/validate_pairs.py --input data/parsed/sillok/articles_with_korean.jsonl --dump-flagged
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# CJK 문자 범위
# ---------------------------------------------------------------------------
CJK_RANGE = (
    r"\u4E00-\u9FFF"
    r"\u3400-\u4DBF"
    r"\uF900-\uFAFF"
    r"\U00020000-\U0002A6DF"
)

# (漢字) 괄호 한자 패턴
ANNOTATION_RE = re.compile(r"\([" + CJK_RANGE + r"·]+\)")

# term(漢字) 패턴 — 한글 term + (漢字) 캡처
TERM_ANNOTATION_RE = re.compile(
    r"([가-힣]{1,15})"           # 한글 term
    r"(\([" + CJK_RANGE + r"·]+\))"  # (漢字)
)

# 인라인 각주 패턴: (漢字) 짧은설명.
INLINE_NOTE_RE = re.compile(
    r"\([" + CJK_RANGE + r"·]+\)"
    r" (?P<note>[^\"\"\"\n]{1,50}?)\."
    r'(?=[ \"\"\"\n]|$)'
)

# trailing space 패턴: (漢字) 뒤에 공백 + 조사/조사 시작 문자
TRAILING_SPACE_RE = re.compile(
    r"\([" + CJK_RANGE + r"·]+\)"
    r" (?=[가-힣])"   # 공백 + 한글 (조사나 다음 어절)
)

# 조사 패턴: trailing space 뒤 조사인지 확인
PARTICLE_AFTER_PAREN_RE = re.compile(
    r"(\([" + CJK_RANGE + r"·]+\))"
    r" (이|가|은|는|을|를|에|에서|의|와|과|으로|로|에게|도|만|까지|부터"
    r"|보다|처럼|같이|에는|에도|으로서|로서|으로써|로써|이며|이고"
    r"|이지만|이라|이라고|라고)"
    r"(?=[^가-힣]|$)"
)


def validate_article(
    article: dict,
    processed: bool = False,
) -> dict[str, list[str]]:
    """단일 기사의 품질 문제를 탐지한다.

    Args:
        article: 기사 dict (original, translation 필드 필수)
        processed: True면 clean_pairs.jsonl 포맷 (variant 필드 있음)

    Returns:
        {"issue_type": [상세 메시지, ...]} 형태의 dict. 문제 없으면 빈 dict.
    """
    issues: dict[str, list[str]] = defaultdict(list)

    original = article.get("original", "")
    translation = article.get("translation", "")

    if not translation:
        return dict(issues)

    # ----- 1. 한자 불일치: 번역의 (漢字)가 원문에 없음 -----
    for m in ANNOTATION_RE.finditer(translation):
        hanja = m.group()[1:-1]  # 괄호 제거
        # 원문에 해당 한자가 포함되어 있는지 확인
        # 중간점(·) 구분된 경우 각각 확인
        chars_to_check = hanja.replace("·", "")
        missing = [c for c in chars_to_check if c not in original]
        if len(missing) > len(chars_to_check) * 0.5:
            issues["hanzi_mismatch"].append(
                f"({hanja}) → 원문에 50%+ 미포함: {''.join(missing[:5])}"
            )

    # ----- 2. 중복 용어: 동일 term(漢字)가 2회+ 등장 -----
    term_counts: Counter[str] = Counter()
    for m in TERM_ANNOTATION_RE.finditer(translation):
        full = m.group(1) + m.group(2)  # term(漢字)
        term_counts[full] += 1

    for term, count in term_counts.items():
        if count >= 2:
            issues["duplicate_term"].append(
                f"{term} ×{count}"
            )

    # ----- 3. 인라인 각주 잔존: (漢字) 짧은설명. -----
    for m in INLINE_NOTE_RE.finditer(translation):
        note = m.group("note").strip()
        # 본문 계속이 아닌 실제 각주인지 간단히 필터
        first_word = note.split()[0] if note.split() else ""
        continuation_words = {
            "이", "가", "은", "는", "을", "를", "에", "에서",
            "의", "와", "과", "으로", "로", "에게", "도",
        }
        if first_word not in continuation_words and len(note) <= 30:
            issues["inline_note"].append(
                f"...{m.group()[:60]}"
            )

    # ----- 4. trailing space: (漢字) 조사 -----
    for m in PARTICLE_AFTER_PAREN_RE.finditer(translation):
        issues["trailing_space"].append(
            f"{m.group()[:40]}"
        )

    return dict(issues)


def king_from_article(article: dict) -> str:
    """기사에서 왕대 정보를 추출한다."""
    return article.get("king", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터 정합성 검증")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="검증할 JSONL 파일",
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="clean_pairs.jsonl 포맷 (variant 필드 있음)",
    )
    parser.add_argument(
        "--dump-flagged",
        action="store_true",
        help="플래그된 기사를 별도 JSONL로 저장",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="검증할 최대 기사 수 (0=전체)",
    )
    parser.add_argument(
        "--article-id",
        type=str,
        default="",
        help="특정 기사 ID만 검증 (디버그용)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"[ERROR] 파일 없음: {input_path}")
        return

    # 왕대별 통계
    king_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    total_stats: dict[str, int] = defaultdict(int)
    flagged_articles: list[dict] = []
    total_articles = 0
    total_with_translation = 0

    print(f"[INFO] 검증 시작: {input_path}")
    print(f"[INFO] 모드: {'processed' if args.processed else 'raw'}")
    print()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            total_articles += 1

            if args.article_id and article.get("article_id") != args.article_id:
                continue

            if not article.get("translation"):
                continue

            total_with_translation += 1
            king = king_from_article(article)

            issues = validate_article(article, processed=args.processed)

            if issues:
                flagged_articles.append({
                    "article_id": article.get("article_id", ""),
                    "king": king,
                    "issues": issues,
                })

                for issue_type, details in issues.items():
                    count = len(details)
                    king_stats[king][issue_type] += count
                    total_stats[issue_type] += count

            if args.limit > 0 and total_with_translation >= args.limit:
                break

    # 결과 출력
    print("=" * 70)
    print(f"검증 완료: {total_articles:,}건 중 번역 있는 {total_with_translation:,}건 검사")
    print(f"플래그된 기사: {len(flagged_articles):,}건")
    print("=" * 70)

    # 전체 통계
    print("\n[전체 통계]")
    issue_labels = {
        "hanzi_mismatch": "한자 불일치",
        "duplicate_term": "중복 용어",
        "inline_note": "인라인 각주",
        "trailing_space": "trailing space",
    }
    for issue_type in ["hanzi_mismatch", "duplicate_term", "inline_note", "trailing_space"]:
        count = total_stats.get(issue_type, 0)
        label = issue_labels.get(issue_type, issue_type)
        print(f"  {label}: {count:,}건")

    # 왕대별 통계
    print("\n[왕대별 통계]")
    all_kings = sorted(king_stats.keys())
    if all_kings:
        header = f"{'왕대':<8}"
        for it in ["hanzi_mismatch", "duplicate_term", "inline_note", "trailing_space"]:
            header += f" {issue_labels[it]:>12}"
        print(header)
        print("-" * len(header))

        for king in all_kings:
            row = f"{king:<8}"
            for it in ["hanzi_mismatch", "duplicate_term", "inline_note", "trailing_space"]:
                row += f" {king_stats[king].get(it, 0):>12,}"
            print(row)

    # 플래그된 기사 상세 (상위 20건)
    if flagged_articles:
        print(f"\n[플래그된 기사 샘플 (상위 20건)]")
        for fa in flagged_articles[:20]:
            try:
                print(f"\n  {fa['article_id']} ({fa['king']})")
                for issue_type, details in fa["issues"].items():
                    label = issue_labels.get(issue_type, issue_type)
                    for d in details[:3]:
                        print(f"    [{label}] {d}")
                    if len(details) > 3:
                        print(f"    ... +{len(details) - 3}건")
            except UnicodeEncodeError:
                print(f"\n  {fa['article_id']} (출력 인코딩 오류, --dump-flagged 사용)")

    # 플래그된 기사 덤프
    if args.dump_flagged and flagged_articles:
        dump_path = input_path.parent / (input_path.stem + "_flagged.jsonl")
        with open(dump_path, "w", encoding="utf-8") as df:
            for fa in flagged_articles:
                df.write(json.dumps(fa, ensure_ascii=False) + "\n")
        print(f"\n[INFO] 플래그 목록 저장: {dump_path} ({len(flagged_articles):,}건)")


if __name__ == "__main__":
    main()
