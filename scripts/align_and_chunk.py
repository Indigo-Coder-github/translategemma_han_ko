"""데이터 전처리 Step 2: 문장 정렬 및 sliding window 청킹.

clean_pairs.jsonl에서 긴 기사(2048 토큰 초과)를 문장 단위로 분할하고
sliding window로 청크를 생성한다. 짧은 기사는 그대로 통과.

처리 순서:
1. 토크나이저 로드 (없으면 문자 수 기반 추정)
2. 각 레코드의 토큰 수 계산
3. max_tokens 이하: 그대로 통과
4. max_tokens 초과: 문장 분할 → 정렬 → sliding window 청킹

Usage:
    python scripts/align_and_chunk.py
    python scripts/align_and_chunk.py --max-tokens 2048 --chunk-size 4 --overlap 2
    python scripts/align_and_chunk.py --model google/translategemma-4b-it --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Instruction 템플릿 (토큰 수 계산에 사용)
# ---------------------------------------------------------------------------
INSTRUCTIONS: dict[str, str] = {
    "plain": "다음 조선시대 한문을 현대 한국어로 번역하라.",
    "annotated": "다음 조선시대 한문을 한자를 병기하여 현대 한국어로 번역하라.",
    "mixed": "다음 조선시대 한문을 필요한 부분에만 한자를 병기하여 현대 한국어로 번역하라.",
}

# 프롬프트 오버헤드 (특수 토큰 + 템플릿 구조)


# ========================== 토크나이저 / 토큰 수 ===========================


def load_tokenizer(model_name: str):
    """HuggingFace 토크나이저를 로드한다. 없으면 다운로드."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[INFO] 토크나이저 로드 완료: {model_name}")
    return tokenizer


def count_tokens(
    original: str,
    translation: str,
    instruction_type: str,
    tokenizer,
    context_original: str | None = None,
    context_translation: str | None = None,
) -> int:
    """instruction + original + translation의 토큰 수를 계산한다."""
    instruction = INSTRUCTIONS.get(instruction_type, INSTRUCTIONS["plain"])

    parts = [instruction, original, translation]
    if context_original:
        parts.append(context_original)
    if context_translation:
        parts.append(context_translation)

    full_text = "\n\n".join(parts)
    return len(tokenizer.encode(full_text))


# ============================= 문장 분할 ==================================


def split_original_sentences(text: str) -> list[str]:
    """한문 원문을 문장 단위로 분할한다.

    구점(。)을 기준으로 분할하되, 구점을 문장 끝에 붙여서 유지한다.
    """
    # 。기준 분할, 빈 문자열 제거
    parts = re.split(r"(。)", text)

    sentences: list[str] = []
    buf = ""
    for part in parts:
        buf += part
        if part == "。":
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    return [s for s in sentences if s]


def split_korean_sentences(text: str) -> list[str]:
    """한국어 번역을 문장 단위로 분할한다.

    마침표(.), 느낌표(!), 물음표(?) 뒤의 공백을 기준으로 분할.
    따옴표 내부의 문장 부호는 분할하지 않도록 간단히 처리.
    """
    # 문장 종결 부호 + 공백(또는 문자열 끝)으로 분할
    # 각 문장에 종결 부호 포함
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


# ============================= 문장 정렬 ==================================


def align_sentences(
    orig_sents: list[str],
    trans_sents: list[str],
) -> list[tuple[str, str]]:
    """원문과 번역 문장을 정렬한다.

    문장 수가 비슷하면(±30%) 위치 기반 1:1 정렬.
    크게 다르면 길이비 기반 탐욕 정렬.

    Returns:
        [(원문_문장, 번역_문장), ...] 정렬된 쌍 리스트
    """
    if not orig_sents or not trans_sents:
        return []

    n_orig = len(orig_sents)
    n_trans = len(trans_sents)

    # 1:1 또는 유사 비율이면 위치 기반 정렬
    if n_orig == n_trans:
        return list(zip(orig_sents, trans_sents))

    ratio = n_trans / n_orig if n_orig > 0 else 1.0

    if 0.7 <= ratio <= 1.3:
        # 비슷한 수: 짧은 쪽에 맞추어 1:1 정렬, 나머지는 마지막에 합침
        pairs: list[tuple[str, str]] = []
        min_n = min(n_orig, n_trans)
        for i in range(min_n - 1):
            pairs.append((orig_sents[i], trans_sents[i]))
        # 마지막 문장에 나머지 합침
        remaining_orig = " ".join(orig_sents[min_n - 1 :])
        remaining_trans = " ".join(trans_sents[min_n - 1 :])
        pairs.append((remaining_orig, remaining_trans))
        return pairs

    # 문장 수 차이가 큰 경우: 길이비 기반 탐욕 정렬
    return _greedy_align(orig_sents, trans_sents)


def _greedy_align(
    orig_sents: list[str],
    trans_sents: list[str],
) -> list[tuple[str, str]]:
    """길이비 기반 탐욕 정렬.

    원문:번역 길이비(~2.3x)를 기준으로 원문 한 문장에
    번역 N 문장을 대응시킨다.
    """
    EXPANSION_RATIO = 2.3
    pairs: list[tuple[str, str]] = []

    ti = 0  # 번역 문장 인덱스
    for oi, orig_sent in enumerate(orig_sents):
        expected_len = len(orig_sent) * EXPANSION_RATIO

        # 남은 원문 문장 수에 비례하여 번역 문장 할당
        remaining_orig = len(orig_sents) - oi
        remaining_trans = len(trans_sents) - ti
        target_trans_count = max(1, round(remaining_trans / remaining_orig))

        # 번역 문장 모으기
        collected: list[str] = []
        collected_len = 0
        for j in range(target_trans_count):
            if ti + j >= len(trans_sents):
                break
            collected.append(trans_sents[ti + j])
            collected_len += len(trans_sents[ti + j])

        # 길이가 부족하면 더 모으기
        while (
            collected_len < expected_len * 0.5
            and ti + len(collected) < len(trans_sents)
            and oi < len(orig_sents) - 1
        ):
            collected.append(trans_sents[ti + len(collected)])
            collected_len += len(collected[-1])

        if not collected:
            collected = [trans_sents[ti]] if ti < len(trans_sents) else [""]

        pairs.append((orig_sent, " ".join(collected)))
        ti += len(collected)

    # 남은 번역 문장을 마지막 쌍에 합침
    if ti < len(trans_sents):
        leftover = " ".join(trans_sents[ti:])
        if pairs:
            last_orig, last_trans = pairs[-1]
            pairs[-1] = (last_orig, last_trans + " " + leftover)
        else:
            pairs.append(("", leftover))

    return pairs


# ========================= sliding window 청킹 ============================


def chunk_aligned_pairs(
    aligned: list[tuple[str, str]],
    instruction_type: str,
    tokenizer,
    max_tokens: int = 2048,
    target_chunk_size: int = 4,
    overlap: int = 2,
) -> list[dict]:
    """정렬된 문장쌍을 sliding window로 청킹한다.

    Args:
        aligned: [(원문, 번역), ...] 정렬된 문장쌍
        instruction_type: instruction 종류 (토큰 수 계산용)
        tokenizer: 토크나이저 (None이면 문자 수 추정)
        max_tokens: 최대 토큰 수
        target_chunk_size: 목표 청크 크기 (문장쌍 수)
        overlap: 이전 청크와의 overlap (문장쌍 수)

    Returns:
        청크 dict 리스트
    """
    chunks: list[dict] = []
    i = 0

    while i < len(aligned):
        # target_chunk_size부터 줄여가며 max_tokens에 맞추기
        for size in range(target_chunk_size, 0, -1):
            chunk_pairs = aligned[i : i + size]
            if not chunk_pairs:
                break

            orig_text = " ".join(o for o, _ in chunk_pairs)
            trans_text = " ".join(t for _, t in chunk_pairs)

            # context (이전 청크의 마지막 문장들)
            ctx_orig = None
            ctx_trans = None
            if i > 0 and overlap > 0:
                ctx_start = max(0, i - overlap)
                ctx_pairs = aligned[ctx_start:i]
                ctx_orig = " ".join(o for o, _ in ctx_pairs)
                ctx_trans = " ".join(t for _, t in ctx_pairs)

            token_count = count_tokens(
                orig_text,
                trans_text,
                instruction_type,
                tokenizer,
                ctx_orig,
                ctx_trans,
            )

            if token_count <= max_tokens:
                chunks.append({
                    "original": orig_text,
                    "translation": trans_text,
                    "context_original": ctx_orig,
                    "context_translation": ctx_trans,
                    "is_first_chunk": i == 0,
                    "token_count": token_count,
                })
                i += size
                break
        else:
            # 문장쌍 1개도 max_tokens 초과 → 그대로 포함 + 경고
            if i < len(aligned):
                orig_text, trans_text = aligned[i]
                ctx_orig = None
                ctx_trans = None
                if i > 0 and overlap > 0:
                    ctx_start = max(0, i - overlap)
                    ctx_pairs = aligned[ctx_start:i]
                    ctx_orig = " ".join(o for o, _ in ctx_pairs)
                    ctx_trans = " ".join(t for _, t in ctx_pairs)

                token_count = count_tokens(
                    orig_text, trans_text, instruction_type,
                    tokenizer, ctx_orig, ctx_trans,
                )
                chunks.append({
                    "original": orig_text,
                    "translation": trans_text,
                    "context_original": ctx_orig,
                    "context_translation": ctx_trans,
                    "is_first_chunk": i == 0,
                    "token_count": token_count,
                })
                i += 1

    return chunks


# ================================= main ====================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="문장 정렬 및 sliding window 청킹",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/sillok/clean_pairs.jsonl"),
        help="입력 JSONL (default: data/processed/sillok/clean_pairs.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sillok/chunked_pairs.jsonl"),
        help="출력 JSONL (default: data/processed/sillok/chunked_pairs.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/translategemma-4b-it",
        help="HF 모델 (토크나이저용, default: google/translategemma-4b-it)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="최대 토큰 수 (default: 2048)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="목표 청크 크기: 문장쌍 수 (default: 4)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="overlap 문장쌍 수 (default: 2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="처리할 최대 기사 수 (0=전체, 테스트용)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 토크나이저 로드
    tokenizer = load_tokenizer(args.model)

    # 통계
    pass_through = 0
    chunked_articles = 0
    total_chunks = 0
    over_limit_singles = 0
    output_count = 0

    print(f"[INFO] 입력: {input_path}")
    print(f"[INFO] max_tokens={args.max_tokens}, chunk_size={args.chunk_size}, overlap={args.overlap}")

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, 1):
            record = json.loads(line)

            if args.limit > 0 and line_no > args.limit:
                break

            article_id = record["article_id"]
            original = record["original"]
            translation = record["translation"]
            instruction_type = record.get("instruction_type", "plain")

            # 토큰 수 계산
            token_count = count_tokens(
                original, translation, instruction_type, tokenizer
            )

            if token_count <= args.max_tokens:
                # 그대로 통과
                out = {
                    **record,
                    "chunk_id": f"{article_id}_c000",
                    "context_original": None,
                    "context_translation": None,
                    "is_first_chunk": True,
                    "total_chunks": 1,
                    "token_count": token_count,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                pass_through += 1
                output_count += 1
            else:
                # 문장 분할 + 정렬 + 청킹
                orig_sents = split_original_sentences(original)
                trans_sents = split_korean_sentences(translation)

                # 문장 분할이 안 되면 (구점 없는 경우 등) 통째로 통과
                if len(orig_sents) <= 1 and len(trans_sents) <= 1:
                    out = {
                        **record,
                        "chunk_id": f"{article_id}_c000",
                        "context_original": None,
                        "context_translation": None,
                        "is_first_chunk": True,
                        "total_chunks": 1,
                        "token_count": token_count,
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    over_limit_singles += 1
                    output_count += 1
                    continue

                aligned = align_sentences(orig_sents, trans_sents)
                chunks = chunk_aligned_pairs(
                    aligned,
                    instruction_type,
                    tokenizer,
                    args.max_tokens,
                    args.chunk_size,
                    args.overlap,
                )

                chunked_articles += 1
                total_chunks += len(chunks)

                for ci, chunk in enumerate(chunks):
                    out = {
                        "source": record.get("source", "sillok"),
                        "article_id": article_id,
                        "chunk_id": f"{article_id}_c{ci:03d}",
                        "king_code": record["king_code"],
                        "king": record["king"],
                        "original": chunk["original"],
                        "translation": chunk["translation"],
                        "variant": record.get("variant", "clean"),
                        "instruction_type": instruction_type,
                        "context_original": chunk["context_original"],
                        "context_translation": chunk["context_translation"],
                        "is_first_chunk": chunk["is_first_chunk"],
                        "total_chunks": len(chunks),
                        "token_count": chunk["token_count"],
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    output_count += 1

            if line_no % 10_000 == 0:
                print(
                    f"  [{line_no:,}] 통과={pass_through:,} "
                    f"청킹={chunked_articles:,}({total_chunks:,}청크)"
                )

    avg_chunks = total_chunks / chunked_articles if chunked_articles > 0 else 0
    print(f"\n[DONE] 출력: {output_count:,}건 → {output_path}")
    print(f"  그대로 통과: {pass_through:,}건")
    print(f"  청킹 대상: {chunked_articles:,}건 → {total_chunks:,}청크 (평균 {avg_chunks:.1f})")
    if over_limit_singles:
        print(f"  분할 불가 (구점 없음): {over_limit_singles:,}건 (초과 상태로 통과)")


if __name__ == "__main__":
    main()
