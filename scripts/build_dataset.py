"""데이터 전처리 Step 3: 왕대별 분할, instruction 포매팅, 데이터셋 저장.

chunked_pairs.jsonl을 train/val/test로 분할하고
Gemma 3 턴 구조의 instruction 템플릿을 적용하여 최종 데이터셋을 생성한다.

분할 기준 (왕대):
    - train: 태조 ~ 성종  (aa, ba, ca, da, ea, fa, ga, ha, ia)
    - val:   연산군 ~ 명종 (ja, ka, la, lb, ma)
    - test:  선조 ~ 철종   (na, nb, oa, ob, pa, qa, ra, rb, sa, sb, ta, tb, ua, va, wa, xa, ya)
    - 제외:  고종, 순종     (za, zb, zc)

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --save-hf
    python scripts/build_dataset.py --output-dir data/splits --limit 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 왕대 → split 매핑
# ---------------------------------------------------------------------------
TRAIN_KINGS = frozenset({
    "aa",  # 태조
    "ba",  # 정종
    "ca",  # 태종
    "da",  # 세종
    "ea",  # 문종
    "fa",  # 단종
    "ga",  # 세조
    "ha",  # 예종
    "ia",  # 성종
})

VAL_KINGS = frozenset({
    "ja",  # 연산군
    "ka",  # 중종
    "la",  # 인종
    "lb",  # (인종 관련 별도 코드가 있으면)
    "ma",  # 명종
})

TEST_KINGS = frozenset({
    "na",  # 선조
    "nb",  # 선조수정
    "oa",  # 광해군(중초본)
    "ob",  # 광해군(정초본)
    "pa",  # 인조
    "qa",  # 효종
    "ra",  # 현종
    "rb",  # 현종개수
    "sa",  # 숙종
    "sb",  # 숙종보궐정오
    "ta",  # 경종
    "tb",  # 경종수정
    "ua",  # 영조
    "va",  # 정조
    "wa",  # 순조
    "xa",  # 헌종
    "ya",  # 철종
})

EXCLUDED_KINGS = frozenset({"za", "zb", "zc"})  # 고종, 순종, 순종부록

# ---------------------------------------------------------------------------
# Instruction 템플릿
# ---------------------------------------------------------------------------
INSTRUCTIONS: dict[str, str] = {
    "plain": "다음 조선시대 한문을 현대 한국어로 번역하라.",
    "annotated": "다음 조선시대 한문을 한자를 병기하여 현대 한국어로 번역하라.",
    "mixed": "다음 조선시대 한문을 필요한 부분에만 한자를 병기하여 현대 한국어로 번역하라.",
}


# ========================== 포매팅 함수 ====================================


def get_split(king_code: str) -> str | None:
    """왕 코드에서 split 이름을 반환한다. 제외 대상이면 None."""
    if king_code in TRAIN_KINGS:
        return "train"
    elif king_code in VAL_KINGS:
        return "val"
    elif king_code in TEST_KINGS:
        return "test"
    elif king_code in EXCLUDED_KINGS:
        return None
    else:
        # 알 수 없는 코드 → 경고 후 None
        return None


def format_prompt(
    instruction: str,
    original: str,
    context_original: str | None = None,
    context_translation: str | None = None,
) -> str:
    """Gemma 3 턴 구조의 user 프롬프트를 생성한다.

    context가 있으면 [맥락] 섹션을 추가한다.
    """
    parts: list[str] = [instruction]

    if context_original and context_translation:
        parts.append("")
        parts.append("[맥락 - 이전 문장]")
        parts.append(f"원문: {context_original}")
        parts.append(f"번역: {context_translation}")
        parts.append("")
        parts.append("[번역할 문장]")

    parts.append("")
    parts.append(original)

    return "\n".join(parts)


def format_gemma3(user_text: str, model_text: str) -> str:
    """Gemma 3 턴 구조의 전체 텍스트를 생성한다."""
    return (
        f"<bos><start_of_turn>user\n"
        f"{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{model_text}<end_of_turn>"
    )


# ================================= main ====================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="왕대별 train/val/test 분할 및 instruction 포매팅",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/sillok/chunked_pairs.jsonl"),
        help="입력 JSONL (default: data/processed/sillok/chunked_pairs.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="출력 디렉토리 (default: data/splits)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/translategemma-4b-it",
        help="HF 모델 (토크나이저용, default: google/translategemma-4b-it)",
    )
    parser.add_argument(
        "--save-hf",
        action="store_true",
        help="HuggingFace DatasetDict (arrow) 형식으로도 저장",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="처리할 최대 레코드 수 (0=전체, 테스트용)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 토크나이저 (토큰 수 재계산용)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[INFO] 토크나이저 로드: {args.model}")

    # split별 파일 핸들
    split_paths = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }
    split_files = {
        name: open(path, "w", encoding="utf-8")
        for name, path in split_paths.items()
    }

    # 통계
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_token_sums: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    variant_per_split: dict[str, dict[str, int]] = {
        s: {"clean": 0, "annotated": 0, "mixed": 0} for s in split_counts
    }
    excluded_count = 0
    unknown_king_codes: set[str] = set()

    print(f"[INFO] 입력: {input_path}")
    print(f"[INFO] 출력: {output_dir}")

    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, 1):
                if args.limit > 0 and line_no > args.limit:
                    break

                record = json.loads(line)
                king_code = record["king_code"]

                split = get_split(king_code)
                if split is None:
                    if king_code not in EXCLUDED_KINGS:
                        unknown_king_codes.add(king_code)
                    excluded_count += 1
                    continue

                # instruction 선택
                instruction_type = record.get("instruction_type", "plain")
                instruction = INSTRUCTIONS.get(instruction_type, INSTRUCTIONS["plain"])

                # 프롬프트 포매팅
                user_text = format_prompt(
                    instruction,
                    record["original"],
                    record.get("context_original"),
                    record.get("context_translation"),
                )
                model_text = record["translation"]
                formatted = format_gemma3(user_text, model_text)

                # 토큰 수 재계산
                token_count = len(tokenizer.encode(formatted))

                out = {
                    "article_id": record["article_id"],
                    "chunk_id": record.get("chunk_id", record["article_id"] + "_c000"),
                    "king_code": king_code,
                    "split": split,
                    "instruction": instruction,
                    "input": record["original"],
                    "output": model_text,
                    "context_input": record.get("context_original"),
                    "context_output": record.get("context_translation"),
                    "formatted": formatted,
                    "token_count": token_count,
                    "variant": record.get("variant", "clean"),
                }

                split_files[split].write(
                    json.dumps(out, ensure_ascii=False) + "\n"
                )
                split_counts[split] += 1
                split_token_sums[split] += token_count

                variant = record.get("variant", "clean")
                if variant in variant_per_split[split]:
                    variant_per_split[split][variant] += 1

                if line_no % 10_000 == 0:
                    total = sum(split_counts.values())
                    print(f"  [{line_no:,}] train={split_counts['train']:,} "
                          f"val={split_counts['val']:,} test={split_counts['test']:,}")

    finally:
        for fh in split_files.values():
            fh.close()

    # ------------------------------------------------------------------
    # 통계 출력
    # ------------------------------------------------------------------
    total = sum(split_counts.values())
    print(f"\n[DONE] 총 {total:,}건 분할 완료")
    if excluded_count:
        print(f"  제외: {excluded_count:,}건 (고종/순종 등)")
    if unknown_king_codes:
        print(f"  [WARN] 알 수 없는 왕 코드: {unknown_king_codes}")

    for split_name in ["train", "val", "test"]:
        count = split_counts[split_name]
        if count == 0:
            print(f"\n  {split_name}: 0건 (데이터 없음)")
            continue

        avg_tokens = split_token_sums[split_name] / count
        print(f"\n  {split_name}: {count:,}건 (평균 {avg_tokens:.0f} tokens)")
        print(f"    → {split_paths[split_name]}")

        vd = variant_per_split[split_name]
        for v in ["clean", "annotated", "mixed"]:
            vc = vd.get(v, 0)
            pct = vc / count * 100 if count > 0 else 0
            print(f"    {v}: {vc:,}건 ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # HuggingFace DatasetDict 저장 (선택)
    # ------------------------------------------------------------------
    if args.save_hf:
        try:
            from datasets import Dataset, DatasetDict

            hf_dir = output_dir / "hf_dataset"
            print(f"\n[INFO] HuggingFace DatasetDict 저장: {hf_dir}")

            splits_dict = {}
            for split_name, path in split_paths.items():
                if split_counts[split_name] == 0:
                    continue

                records = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        records.append(json.loads(line))

                splits_dict[split_name] = Dataset.from_list(records)
                print(f"  {split_name}: {len(records):,}건 → Dataset")

            if splits_dict:
                ds = DatasetDict(splits_dict)
                ds.save_to_disk(str(hf_dir))
                print(f"[DONE] HuggingFace DatasetDict → {hf_dir}")
            else:
                print("[WARN] 저장할 데이터가 없습니다.")

        except ImportError:
            print("[ERROR] datasets 라이브러리가 설치되지 않았습니다: pip install datasets")
        except Exception as e:
            print(f"[ERROR] HF DatasetDict 저장 실패: {e}")


if __name__ == "__main__":
    main()
