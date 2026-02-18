"""데이터 전처리 Step 3: 랜덤 분할, instruction 포매팅, 데이터셋 저장.

chunked_pairs.jsonl을 train/val/test로 분할하고
Gemma 3 턴 구조의 instruction 템플릿을 적용하여 최종 데이터셋을 생성한다.

분할 기준:
    - 랜덤 셔플 후 80:10:10 (seed 고정으로 재현 가능)
    - 제외: 고종, 순종 (za, zb, zc) — 근대 문체 전환기, 일본어·서양 용어 혼용

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --save-hf
    python scripts/build_dataset.py --seed 42 --train-ratio 0.8 --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# 제외 대상
# ---------------------------------------------------------------------------
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
        description="랜덤 train/val/test 분할 및 instruction 포매팅",
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
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="train 비율 (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="val 비율 (default: 0.1)",
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

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(f"train_ratio + val_ratio > 1.0: {train_ratio} + {val_ratio}")

    # 토크나이저 (토큰 수 재계산용)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[INFO] 토크나이저 로드: {args.model}")
    print(f"[INFO] 입력: {input_path}")
    print(f"[INFO] 분할 비율: train={train_ratio:.0%} val={val_ratio:.0%} test={test_ratio:.0%}")
    print(f"[INFO] seed={args.seed}")

    # ------------------------------------------------------------------
    # 1단계: 전체 레코드 로드 + 포매팅
    # ------------------------------------------------------------------
    records: list[dict] = []
    excluded_count = 0

    print("[INFO] 레코드 로드 및 포매팅 중...")
    with open(input_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if args.limit > 0 and line_no > args.limit:
                break

            record = json.loads(line)
            king_code = record["king_code"]

            if king_code in EXCLUDED_KINGS:
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

            records.append({
                "article_id": record["article_id"],
                "chunk_id": record.get("chunk_id", record["article_id"] + "_c000"),
                "king_code": king_code,
                "instruction": instruction,
                "input": record["original"],
                "output": model_text,
                "context_input": record.get("context_original"),
                "context_output": record.get("context_translation"),
                "formatted": formatted,
                "token_count": token_count,
                "variant": record.get("variant", "clean"),
            })

            if line_no % 10_000 == 0:
                print(f"  [{line_no:,}] 로드 완료 (유효 {len(records):,}건)")

    print(f"[INFO] 총 {len(records):,}건 로드 (제외 {excluded_count:,}건)")

    # ------------------------------------------------------------------
    # 2단계: 셔플 + 분할
    # ------------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split_map = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    # ------------------------------------------------------------------
    # 3단계: 파일 출력
    # ------------------------------------------------------------------
    split_paths = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }

    split_counts: dict[str, int] = {}
    split_token_sums: dict[str, int] = {}
    variant_per_split: dict[str, dict[str, int]] = {}

    for split_name, split_records in split_map.items():
        path = split_paths[split_name]
        count = 0
        token_sum = 0
        variant_counts: dict[str, int] = {"clean": 0, "annotated": 0, "mixed": 0}

        with open(path, "w", encoding="utf-8") as fout:
            for rec in split_records:
                rec["split"] = split_name
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                token_sum += rec["token_count"]
                v = rec.get("variant", "clean")
                if v in variant_counts:
                    variant_counts[v] += 1

        split_counts[split_name] = count
        split_token_sums[split_name] = token_sum
        variant_per_split[split_name] = variant_counts

    # ------------------------------------------------------------------
    # 통계 출력
    # ------------------------------------------------------------------
    total = sum(split_counts.values())
    print(f"\n[DONE] 총 {total:,}건 분할 완료")
    if excluded_count:
        print(f"  제외: {excluded_count:,}건 (고종/순종)")

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
