"""파이프라인 실행 결과 요약.

파이프라인 출력 파일을 읽어 CLAUDE.md에 붙여넣을 수 있는
통계 블록(마크다운)을 생성한다.

Usage:
    python scripts/pipeline_summary.py
    python scripts/pipeline_summary.py --update-claude  # CLAUDE.md 자동 갱신
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# 파일 경로 기본값
# ---------------------------------------------------------------------------
ARTICLES_WITH_KOREAN = Path("data/parsed/sillok/articles_with_korean.jsonl")
CLEAN_PAIRS = Path("data/processed/sillok/clean_pairs.jsonl")
CHUNKED_PAIRS = Path("data/processed/sillok/chunked_pairs.jsonl")
TRAIN_SPLIT = Path("data/splits/train.jsonl")
VAL_SPLIT = Path("data/splits/val.jsonl")
TEST_SPLIT = Path("data/splits/test.jsonl")


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def count_lines(path: Path) -> int:
    """파일 줄 수를 반환한다."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_jsonl(path: Path):
    """JSONL 파일을 한 줄씩 yield한다."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Step 1 통계: prepare_pairs.py 출력 분석
# ---------------------------------------------------------------------------

def summarize_step1() -> dict:
    if not ARTICLES_WITH_KOREAN.exists() or not CLEAN_PAIRS.exists():
        return {}

    input_count = count_lines(ARTICLES_WITH_KOREAN)

    no_translation = sum(
        1 for r in read_jsonl(ARTICLES_WITH_KOREAN)
        if not r.get("translation")
    )

    variant_counts: Counter = Counter()
    note_removed_approx = 0
    output_count = 0

    for r in read_jsonl(CLEAN_PAIRS):
        output_count += 1
        variant_counts[r.get("variant", "clean")] += 1

    duplicates_removed = (input_count - no_translation) - output_count

    return {
        "input_count": input_count,
        "no_translation": no_translation,
        "duplicates_removed": max(duplicates_removed, 0),
        "output_count": output_count,
        "variants": dict(variant_counts),
    }


# ---------------------------------------------------------------------------
# validate_pairs.py 결과 분석 (flagged 파일이 있으면)
# ---------------------------------------------------------------------------

def summarize_validate() -> dict:
    flagged_path = Path("data/processed/sillok/clean_pairs_flagged.jsonl")
    if not flagged_path.exists() or not CLEAN_PAIRS.exists():
        return {}

    total = count_lines(CLEAN_PAIRS)
    trailing_space = 0
    hanja_mismatch = 0
    duplicate_term = 0
    inline_note = 0

    for r in read_jsonl(flagged_path):
        flags = r.get("flags", [])
        for f in flags:
            if "trailing" in f:
                trailing_space += 1
            elif "한자 불일치" in f or "hanja" in f.lower():
                hanja_mismatch += 1
            elif "중복" in f or "duplicate" in f.lower():
                duplicate_term += 1
            elif "인라인" in f or "inline" in f.lower():
                inline_note += 1

    flagged_count = count_lines(flagged_path)
    return {
        "total": total,
        "flagged": flagged_count,
        "trailing_space": trailing_space,
        "hanja_mismatch": hanja_mismatch,
        "duplicate_term": duplicate_term,
        "inline_note": inline_note,
    }


# ---------------------------------------------------------------------------
# Step 2 통계: align_and_chunk.py 출력 분석
# ---------------------------------------------------------------------------

def summarize_step2() -> dict:
    if not CHUNKED_PAIRS.exists():
        return {}

    pass_through = 0
    chunked_articles: set = set()
    total_chunks_from_chunked = 0
    output_count = 0
    over_limit = 0

    for r in read_jsonl(CHUNKED_PAIRS):
        output_count += 1
        total_chunks = r.get("total_chunks", 1)
        if total_chunks == 1:
            pass_through += 1
        else:
            chunked_articles.add(r["article_id"])
            total_chunks_from_chunked += 1

    n_chunked = len(chunked_articles)
    n_chunks = total_chunks_from_chunked
    avg = n_chunks / n_chunked if n_chunked > 0 else 0

    return {
        "pass_through": pass_through,
        "chunked_articles": n_chunked,
        "total_chunks": n_chunks,
        "avg_chunks": round(avg, 1),
        "output_count": output_count,
    }


# ---------------------------------------------------------------------------
# Step 3 통계: build_dataset.py 출력 분석
# ---------------------------------------------------------------------------

def summarize_step3() -> dict:
    result = {}
    total = 0
    for split_name, path in [("train", TRAIN_SPLIT), ("val", VAL_SPLIT), ("test", TEST_SPLIT)]:
        if not path.exists():
            continue
        count = 0
        token_sum = 0
        variants: Counter = Counter()
        for r in read_jsonl(path):
            count += 1
            token_sum += r.get("token_count", 0)
            variants[r.get("variant", "clean")] += 1
        avg_tokens = token_sum / count if count else 0
        result[split_name] = {
            "count": count,
            "avg_tokens": round(avg_tokens),
            "variants": dict(variants),
        }
        total += count
    result["total"] = total
    return result


# ---------------------------------------------------------------------------
# 마크다운 렌더링
# ---------------------------------------------------------------------------

def render_markdown(s1: dict, sv: dict, s2: dict, s3: dict, run_date: str) -> str:
    lines = [f"### 파이프라인 실행 결과 ({run_date})"]
    lines.append("")

    # Step 1
    if s1:
        lines.append("**Step 1: prepare_pairs.py**")
        lines.append("")
        lines.append("| 항목 | 수치 |")
        lines.append("|------|------|")
        lines.append(f"| 입력 (articles_with_korean.jsonl) | {s1['input_count']:,}건 |")
        lines.append(f"| 번역 있음 | {s1['input_count'] - s1['no_translation']:,}건 |")
        lines.append(f"| 번역 없음 (제거) | {s1['no_translation']:,}건 |")
        lines.append(f"| 중복 제거 | {s1['duplicates_removed']:,}건 |")
        lines.append(f"| **순수 출력** | **{s1['output_count']:,}건** |")
        for v in ["clean", "annotated", "mixed"]:
            c = s1["variants"].get(v, 0)
            pct = c / s1["output_count"] * 100 if s1["output_count"] else 0
            lines.append(f"| variant: {v} | {c:,} ({pct:.1f}%) |")
        lines.append("")

    # Validate
    if sv:
        lines.append("**데이터 검증: validate_pairs.py**")
        lines.append("")
        lines.append("| 검증 항목 | 건수 | 비고 |")
        lines.append("|-----------|------|------|")
        lines.append(f"| trailing space | {sv.get('trailing_space', 0):,} | |")
        lines.append(f"| 한자 불일치 | {sv.get('hanja_mismatch', 0):,} | 번역의 `(漢字)`가 원문에 미포함 |")
        lines.append(f"| 중복 용어 | {sv.get('duplicate_term', 0):,} | 동일 인물/용어 반복 언급 (대부분 정상) |")
        lines.append(f"| 인라인 각주 | {sv.get('inline_note', 0):,} | `content` 폴백 기사에서 주로 발생 |")
        flagged = sv.get("flagged", 0)
        total = sv.get("total", 0)
        pct = flagged / total * 100 if total else 0
        lines.append(f"| 총 플래그 기사 | {flagged:,} ({pct:.1f}%) | |")
        lines.append("")

    # Step 2
    if s2:
        lines.append("**Step 2: align_and_chunk.py**")
        lines.append("")
        lines.append("| 항목 | 수치 |")
        lines.append("|------|------|")
        lines.append(f"| 그대로 통과 (≤2048 tokens) | {s2['pass_through']:,}건 ({s2['pass_through']/s2['output_count']*100:.1f}%) |")
        lines.append(f"| 청킹 대상 (>2048 tokens) | {s2['chunked_articles']:,}건 |")
        lines.append(f"| 생성된 청크 수 | {s2['total_chunks']:,}개 (평균 {s2['avg_chunks']}청크/기사) |")
        lines.append(f"| **총 출력** | **{s2['output_count']:,}건** |")
        lines.append("")

    # Step 3
    if s3:
        lines.append("**Step 3: build_dataset.py (랜덤 80:10:10, seed=42)**")
        lines.append("")
        lines.append("| Split | 건수 | 비율 | 평균 tokens | clean | annotated | mixed |")
        lines.append("|-------|------|------|------------|-------|-----------|-------|")
        total = s3.get("total", 0)
        for split_name in ["train", "val", "test"]:
            s = s3.get(split_name)
            if not s:
                continue
            count = s["count"]
            ratio = count / total * 100 if total else 0
            v = s["variants"]
            def pct(vname):
                c = v.get(vname, 0)
                return f"{c:,} ({c/count*100:.1f}%)" if count else "0"
            lines.append(
                f"| {split_name} | {count:,} | {ratio:.1f}% | {s['avg_tokens']} "
                f"| {pct('clean')} | {pct('annotated')} | {pct('mixed')} |"
            )
        lines.append("")
        lines.append("> 수집 미완료 데이터 추가 시 재실행 필요.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLAUDE.md 자동 갱신
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(
    r"(### 파이프라인 실행 결과 \(.+?\)\n)"  # 시작 헤더
    r".*?"                                   # 내용 (non-greedy)
    r"(?=\n## |\n### |\Z)",                  # 다음 섹션 또는 파일 끝
    re.DOTALL,
)


def update_claude_md(markdown: str, claude_path: Path) -> None:
    text = claude_path.read_text(encoding="utf-8")
    replacement = markdown + "\n"
    new_text, n = SECTION_RE.subn(replacement, text, count=1)
    if n == 0:
        # 섹션이 없으면 ## 학습 앞에 삽입
        insert_before = "\n## 학습\n"
        if insert_before in new_text:
            new_text = new_text.replace(insert_before, f"\n{replacement}\n## 학습\n", 1)
        else:
            new_text = new_text + "\n" + replacement
    claude_path.write_text(new_text, encoding="utf-8")
    print(f"[INFO] CLAUDE.md 갱신 완료: {claude_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="파이프라인 결과 요약 → CLAUDE.md용 마크다운 생성")
    parser.add_argument(
        "--update-claude",
        action="store_true",
        help="CLAUDE.md의 파이프라인 실행 결과 섹션을 자동으로 갱신",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=str(date.today().strftime("%Y-%m-%d")),
        help="실행 날짜 표기 (default: 오늘)",
    )
    args = parser.parse_args()

    print("[INFO] 파이프라인 출력 파일 분석 중...")
    s1 = summarize_step1()
    sv = summarize_validate()
    s2 = summarize_step2()
    s3 = summarize_step3()

    # 어떤 파일도 없으면 오류
    if not any([s1, s2, s3]):
        print("[ERROR] 분석할 파이프라인 출력 파일이 없습니다.")
        print("  확인: data/processed/sillok/clean_pairs.jsonl")
        print("        data/processed/sillok/chunked_pairs.jsonl")
        print("        data/splits/{train,val,test}.jsonl")
        return

    # 추가 컨텍스트 (왕대 범위)
    king_codes: set = set()
    if CLEAN_PAIRS.exists():
        for r in read_jsonl(CLEAN_PAIRS):
            king_codes.add(r.get("king_code", ""))
    king_range = f"{min(king_codes)}~{max(king_codes)}" if king_codes else "?"

    run_date = f"{args.date}, {king_range}"
    markdown = render_markdown(s1, sv, s2, s3, run_date)

    print("\n" + "=" * 60)
    print(markdown)
    print("=" * 60)

    if args.update_claude:
        claude_path = Path("CLAUDE.md")
        if not claude_path.exists():
            print(f"[ERROR] {claude_path} 를 찾을 수 없습니다.")
            return
        update_claude_md(markdown, claude_path)
    else:
        print("\n[INFO] CLAUDE.md에 반영하려면 --update-claude 옵션을 추가하세요.")


if __name__ == "__main__":
    main()
