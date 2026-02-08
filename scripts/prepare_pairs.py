"""데이터 전처리 Step 1: 필터링, 중복 제거, 역자 주석 제거, 괄호 한자 variant 생성.

수집된 articles_with_korean.jsonl을 정제하여 학습 데이터 쌍을 생성한다.

처리 순서:
1. translation이 없는 레코드 제거
2. (original, translation) 해시 기준 완전 중복 제거
3. 역자 주석 제거 (번역자가 추가한 짧은 설명)
4. 괄호 한자 variant 생성 (clean/annotated/mixed 중 1개 선택)

Usage:
    python scripts/prepare_pairs.py
    python scripts/prepare_pairs.py --clean-ratio 0.5 --annotated-ratio 0.3 --mixed-ratio 0.2
    python scripts/prepare_pairs.py --note-detection strict --limit 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# CJK 문자 범위 (한자 매칭용)
# ---------------------------------------------------------------------------
CJK_RANGE = (
    r"\u4E00-\u9FFF"       # CJK Unified Ideographs
    r"\u3400-\u4DBF"       # CJK Unified Ideographs Extension A
    r"\uF900-\uFAFF"       # CJK Compatibility Ideographs
    r"\U00020000-\U0002A6DF"  # CJK Unified Ideographs Extension B
)

# ---------------------------------------------------------------------------
# 괄호 한자 주석 패턴: (漢字) — 예: (壽昌宮), (裵克廉)
# ---------------------------------------------------------------------------
ANNOTATION_RE = re.compile(r"\([" + CJK_RANGE + r"·]+\)")

# ---------------------------------------------------------------------------
# 역자 주석 후보 패턴
#   (漢字) 짧은설명문.  ← 뒤에 공백·따옴표·줄바꿈 또는 문자열 끝
# ---------------------------------------------------------------------------
NOTE_CANDIDATE_RE = re.compile(
    r"(?P<annotation>\([" + CJK_RANGE + r"·]+\))"  # (漢字) 부분
    r"(?P<note> (?P<note_text>[^\"\"\"\n]{1,50}?)\.)"  # 공백 + 설명 + 마침표
    r'(?=[ \"\"\"\n]|$)'                               # 뒤에 구분자
)

# ---------------------------------------------------------------------------
# 역자 주석 종결 패턴 (strict 모드)
# ---------------------------------------------------------------------------
STRICT_NOTE_ENDINGS = [
    "을 가리킴", "를 가리킴",
    "을 이르는 말", "를 이르는 말",
    "을 말함", "를 말함",
    "을 뜻함", "를 뜻함",
    "을 의미한 것임", "를 의미한 것임",
    "을 의미함", "를 의미함",
    "의 파자",
    "을 점치는 사람", "를 점치는 사람",
    "을 이름", "를 이름",
    "의 별칭", "의 이칭",
    "의 약칭",
]

# ---------------------------------------------------------------------------
# 문장 계속을 나타내는 첫 단어 (조사/어미)
# 이것으로 시작하는 텍스트는 역자 주석이 아니라 본문의 연속
# ---------------------------------------------------------------------------
CONTINUATION_WORDS = frozenset({
    "이", "가", "은", "는", "을", "를", "에", "에서",
    "의", "와", "과", "으로", "로", "에게", "도",
    "만", "까지", "부터", "보다", "처럼", "같이",
    "등", "등이", "등을", "등의", "등에", "등은", "등도",
    "등에서", "등으로", "등과", "등에게", "등은",
    "에는", "에도", "으로서", "로서", "으로써", "로써",
    "이며", "이고", "이지만", "이라", "이라고",
})

# ---------------------------------------------------------------------------
# variant ↔ instruction_type 매핑
# ---------------------------------------------------------------------------
INSTRUCTION_TYPES: dict[str, str] = {
    "clean": "plain",
    "annotated": "annotated",
    "mixed": "mixed",
}


# ============================= 역자 주석 제거 ==============================


def remove_translator_notes(text: str, mode: str = "strict") -> tuple[str, int]:
    """역자 주석을 제거한다.

    역자 주석: 괄호 한자 뒤에 번역자가 추가한 짧은 설명.
    예: ``시좌궁(時坐宮) 그 당시에 왕이 거처하던 궁전.``
         → ``시좌궁(時坐宮)`` (설명 부분만 제거)

    Args:
        text: 번역 텍스트
        mode: ``"strict"`` (높은 정밀도), ``"relaxed"`` (높은 재현율),
              ``"off"`` (제거 안 함)

    Returns:
        (정제된 텍스트, 제거된 주석 수)
    """
    if mode == "off":
        return text, 0

    notes_removed = 0
    matches = list(NOTE_CANDIDATE_RE.finditer(text))

    # 뒤에서부터 제거해야 인덱스가 밀리지 않는다
    for match in reversed(matches):
        note_text = match.group("note_text").strip()
        if _is_note(note_text, mode):
            start = match.start("note")
            end = match.end("note")
            text = text[:start] + text[end:]
            notes_removed += 1

    # 연속 공백 정리
    text = re.sub(r"  +", " ", text)
    return text, notes_removed


def _is_note(text: str, mode: str) -> bool:
    """주어진 텍스트가 역자 주석인지 판별한다."""
    stripped = text.strip()
    if not stripped:
        return False

    # 주석 제거 후 순수 텍스트
    stripped_clean = ANNOTATION_RE.sub("", stripped).strip()
    if not stripped_clean:
        return False

    first_word = stripped_clean.split()[0] if stripped_clean.split() else ""

    if mode == "strict":
        # 1. 알려진 종결 패턴 (가장 확실)
        for ending in STRICT_NOTE_ENDINGS:
            if stripped_clean.endswith(ending):
                return True

        # 2. 매우 짧은 교차 참조 (≤10자, 조사 시작 제외)
        if len(stripped_clean) <= 10 and first_word not in CONTINUATION_WORDS:
            return True

        # 3. 짧은 설명구 (11~20자): 서술형 어미가 없으면 주석으로 판정
        #    예: "그 당시에 왕이 거처하던 궁전" (16자, 명사로 끝남)
        if 10 < len(stripped_clean) <= 20 and first_word not in CONTINUATION_WORDS:
            sentence_endings = [
                "었다", "습니다", "하였다", "되었다", "있었다",
                "하여", "하니", "하므로", "하는데", "었으며",
                "었는데", "하였으며",
            ]
            if not any(ve in stripped_clean for ve in sentence_endings):
                return True

        return False

    elif mode == "relaxed":
        # 1. 너무 긴 텍스트는 주석이 아님
        if len(stripped_clean) > 40:
            return False

        # 2. 조사로 시작하면 본문 연속
        if first_word in CONTINUATION_WORDS:
            return False

        # 3. 서술형 어미로 끝나면 본문
        verb_endings = [
            "었다", "습니다", "하였다", "되었다", "있었다",
            "하니", "하여", "했는데", "인데", "였다",
            "겠다", "었으며", "하였으며",
        ]
        if any(stripped_clean.endswith(ve) for ve in verb_endings):
            return False

        return True

    return False


# ============================ 괄호 한자 처리 ==============================


def remove_annotations(text: str) -> str:
    """모든 괄호 한자 주석을 제거한다 (clean variant).

    예: ``홍언필(洪彦弼)이 아뢰기를`` → ``홍언필이 아뢰기를``
    """
    return ANNOTATION_RE.sub("", text)


def keep_random_annotations(
    text: str,
    keep_ratio: float,
    rng: random.Random,
) -> str:
    """괄호 한자 주석 중 일부만 랜덤으로 유지한다 (mixed variant).

    Args:
        text: 번역 텍스트 (역자 주석 제거 후)
        keep_ratio: 유지 비율 (0.0~1.0)
        rng: 랜덤 생성기

    Returns:
        일부 주석만 남은 텍스트
    """
    matches = list(ANNOTATION_RE.finditer(text))
    if not matches:
        return text

    # 뒤에서부터 처리
    for match in reversed(matches):
        if rng.random() >= keep_ratio:
            text = text[: match.start()] + text[match.end() :]

    return text


# ============================ variant 선택 ================================


def article_to_seed(article_id: str, global_seed: int) -> int:
    """기사 ID에서 결정론적 시드를 생성한다."""
    h = hashlib.md5(f"{article_id}:{global_seed}".encode()).hexdigest()
    return int(h[:8], 16)


def choose_variant(
    article_id: str,
    has_annotations: bool,
    ratios: tuple[float, float, float],
    global_seed: int,
) -> str:
    """기사에 적용할 variant를 결정한다.

    Args:
        article_id: 기사 ID
        has_annotations: 괄호 주석 포함 여부
        ratios: (clean, annotated, mixed) 비율
        global_seed: 글로벌 시드

    Returns:
        ``"clean"``, ``"annotated"``, 또는 ``"mixed"``
    """
    if not has_annotations:
        return "clean"

    seed = article_to_seed(article_id, global_seed)
    rng = random.Random(seed)
    r = rng.random()

    clean_r, annotated_r, _ = ratios
    if r < clean_r:
        return "clean"
    elif r < clean_r + annotated_r:
        return "annotated"
    else:
        return "mixed"


def apply_variant(
    translation: str,
    variant: str,
    article_id: str,
    global_seed: int,
) -> str:
    """번역 텍스트에 variant를 적용한다."""
    if variant == "clean":
        return remove_annotations(translation)
    elif variant == "annotated":
        return translation
    elif variant == "mixed":
        seed = article_to_seed(article_id, global_seed) + 1
        rng = random.Random(seed)
        return keep_random_annotations(translation, keep_ratio=0.4, rng=rng)
    return translation


# ================================= main ====================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="데이터 전처리: 필터링, 중복 제거, 역자 주석 제거, 괄호 한자 variant 생성",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/parsed/sillok/articles_with_korean.jsonl"),
        help="입력 JSONL (default: data/parsed/sillok/articles_with_korean.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sillok/clean_pairs.jsonl"),
        help="출력 JSONL (default: data/processed/sillok/clean_pairs.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (default: 42)",
    )
    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=0.5,
        help="clean variant 비율 (default: 0.5)",
    )
    parser.add_argument(
        "--annotated-ratio",
        type=float,
        default=0.3,
        help="annotated variant 비율 (default: 0.3)",
    )
    parser.add_argument(
        "--mixed-ratio",
        type=float,
        default=0.2,
        help="mixed variant 비율 (default: 0.2)",
    )
    parser.add_argument(
        "--note-detection",
        choices=["strict", "relaxed", "off"],
        default="strict",
        help="역자 주석 탐지 모드 (default: strict)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="처리할 최대 기사 수 (0=전체, 테스트용)",
    )
    args = parser.parse_args()

    # variant 비율 정규화
    ratios = (args.clean_ratio, args.annotated_ratio, args.mixed_ratio)
    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 0.01:
        print(f"[WARN] variant 비율 합계가 1.0이 아닙니다: {total_ratio:.2f}, 정규화합니다.")
        ratios = tuple(r / total_ratio for r in ratios)

    input_path: Path = args.input
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: 번역이 있는 기사만 로드
    # ------------------------------------------------------------------
    print(f"[INFO] 입력 파일 로드: {input_path}")
    articles: list[dict] = []
    total_loaded = 0
    skipped_no_translation = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            total_loaded += 1

            if not article.get("translation"):
                skipped_no_translation += 1
                continue

            articles.append(article)

            if args.limit > 0 and len(articles) >= args.limit:
                break

    print(
        f"[INFO] 로드: {total_loaded:,}건, "
        f"번역 있음: {len(articles):,}건, "
        f"번역 없음: {skipped_no_translation:,}건 제외"
    )

    # ------------------------------------------------------------------
    # Step 2: (original, translation) 해시 기준 완전 중복 제거
    # ------------------------------------------------------------------
    seen: set[str] = set()
    deduped: list[dict] = []
    dup_count = 0

    for article in articles:
        pair_key = article["original"] + "|||" + article["translation"]
        h = hashlib.md5(pair_key.encode("utf-8")).hexdigest()
        if h in seen:
            dup_count += 1
            continue
        seen.add(h)
        deduped.append(article)

    print(f"[INFO] 중복 제거: {dup_count:,}건 → 남은 건수: {len(deduped):,}건")

    # ------------------------------------------------------------------
    # Step 3 & 4: 역자 주석 제거 + variant 생성
    # ------------------------------------------------------------------
    total_notes_removed = 0
    variant_counts: dict[str, int] = {"clean": 0, "annotated": 0, "mixed": 0}
    output_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, article in enumerate(deduped, 1):
            translation = article["translation"]

            # 역자 주석 제거
            cleaned_translation, notes_removed = remove_translator_notes(
                translation, args.note_detection
            )
            total_notes_removed += notes_removed

            # 괄호 주석 존재 여부 확인
            has_annotations = bool(ANNOTATION_RE.search(cleaned_translation))

            # variant 선택
            variant = choose_variant(
                article["article_id"], has_annotations, ratios, args.seed
            )
            variant_counts[variant] += 1

            # variant 적용
            final_translation = apply_variant(
                cleaned_translation, variant, article["article_id"], args.seed
            )

            out = {
                "source": article.get("source", "sillok"),
                "article_id": article["article_id"],
                "king_code": article["king_code"],
                "king": article["king"],
                "original": article["original"],
                "translation": final_translation,
                "variant": variant,
                "instruction_type": INSTRUCTION_TYPES[variant],
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            output_count += 1

            if i % 10_000 == 0 or i == len(deduped):
                print(f"  [{i:,}/{len(deduped):,}] 처리 중...")

    # ------------------------------------------------------------------
    # 통계 출력
    # ------------------------------------------------------------------
    print(f"\n[DONE] 출력: {output_count:,}건 → {output_path}")
    print(f"  역자 주석 제거: {total_notes_removed:,}건 ({args.note_detection} 모드)")
    print(f"  variant 분포:")
    for v, c in variant_counts.items():
        pct = c / output_count * 100 if output_count > 0 else 0
        print(f"    {v}: {c:,}건 ({pct:.1f}%)")


if __name__ == "__main__":
    main()
