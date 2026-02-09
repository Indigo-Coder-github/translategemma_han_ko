"""TranslateGemma LoRA 파인튜닝 스크립트.

HuggingFace Transformers + PEFT를 사용하여 TranslateGemma 모델에
조선시대 한문→한국어 번역 데이터로 LoRA 파인튜닝을 수행한다.

Usage:
    # 단일 GPU (L40s 1장, 12B)
    python training/finetune_lora.py --config training/configs/default.yaml

    # Multi-GPU (L40s 2장) — FSDP2
    accelerate launch training/finetune_lora.py --config training/configs/default.yaml

    # Resume
    python training/finetune_lora.py --config training/configs/default.yaml --resume

    # 로컬 테스트 (4B, 100건)
    python training/finetune_lora.py --config training/configs/default.yaml \
        --model google/translategemma-4b-it --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """YAML 설정 파일을 로드한다."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 데이터 전처리
# ---------------------------------------------------------------------------

LOSS_MASK_LABEL = -100

# <start_of_turn>model\n 이후부터 loss 계산
MODEL_TURN_MARKER = "<start_of_turn>model\n"


def tokenize_and_mask(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict:
    """formatted 텍스트를 토크나이즈하고, user turn에 loss mask를 적용한다.

    model turn (응답) 토큰만 loss를 계산하고,
    그 앞의 시스템/user turn 토큰은 -100으로 마스킹한다.
    """
    text = example["formatted"]

    # 토크나이즈
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )

    input_ids = tokenized["input_ids"]
    labels = list(input_ids)  # 복사

    # model turn 시작 위치 찾기
    # "<start_of_turn>model\n"을 토크나이즈하여 서브토큰 시퀀스를 찾는다
    marker_ids = tokenizer.encode(MODEL_TURN_MARKER, add_special_tokens=False)
    marker_len = len(marker_ids)

    # input_ids에서 marker 시퀀스 위치를 검색
    mask_end = 0  # 기본: 전부 loss 계산
    for i in range(len(input_ids) - marker_len + 1):
        if input_ids[i : i + marker_len] == marker_ids:
            mask_end = i + marker_len
            break

    # mask_end 이전 토큰은 loss 미계산
    for i in range(mask_end):
        labels[i] = LOSS_MASK_LABEL

    tokenized["labels"] = labels
    return tokenized


# ---------------------------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    dtype_str: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """모델과 토크나이저를 로드한다.

    Gemma 3 계열은 bf16 필수 (fp16 → NaN logits).
    """
    dtype = getattr(torch, dtype_str)

    print(f"[INFO] 모델 로드: {model_name} (dtype={dtype_str})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None,  # Trainer가 device placement 관리
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] 모델 파라미터: {model.num_parameters():,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA 설정
# ---------------------------------------------------------------------------

def apply_lora(
    model: AutoModelForCausalLM,
    lora_cfg: dict,
) -> AutoModelForCausalLM:
    """모델에 LoRA를 적용한다."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        use_rslora=lora_cfg.get("use_rslora", True),
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TranslateGemma LoRA 파인튜닝",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/default.yaml"),
        help="설정 파일 경로 (default: training/configs/default.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="모델 ID (config 오버라이드)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="체크포인트 저장 경로 (default: output/<model_name>-lora)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="마지막 체크포인트에서 재개",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="학습 데이터 제한 (0=전체, 테스트용)",
    )
    args = parser.parse_args()

    # 설정 로드
    cfg = load_config(args.config)
    model_name = args.model or cfg["model"]["name"]
    dtype_str = cfg["model"]["dtype"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # 출력 경로
    model_short = model_name.split("/")[-1]
    output_dir = args.output_dir or Path(f"output/{model_short}-lora")

    print(f"[INFO] 설정 파일: {args.config}")
    print(f"[INFO] 모델: {model_name}")
    print(f"[INFO] 출력: {output_dir}")

    # ----- 데이터 로드 -----
    data_path = Path(data_cfg["path"])
    if data_path.exists() and (data_path / "dataset_dict.json").exists():
        print(f"[INFO] HF DatasetDict 로드: {data_path}")
        dataset = load_from_disk(str(data_path))
    else:
        # hf_dataset이 없으면 JSONL에서 로드
        print("[INFO] HF DatasetDict 없음, JSONL에서 로드")
        from datasets import Dataset

        splits_dir = data_path.parent if data_path.name == "hf_dataset" else data_path
        splits = {}
        for split_name in ["train", "val", "test"]:
            jsonl_path = splits_dir / f"{split_name}.jsonl"
            if jsonl_path.exists():
                splits[split_name] = Dataset.from_json(str(jsonl_path))
                print(f"  {split_name}: {len(splits[split_name]):,}건")
        dataset = DatasetDict(splits)

    if "train" not in dataset:
        print("[ERROR] train split이 없습니다.", file=sys.stderr)
        sys.exit(1)

    # limit 적용
    if args.limit > 0:
        for split_name in dataset:
            if len(dataset[split_name]) > args.limit:
                dataset[split_name] = dataset[split_name].select(range(args.limit))
        print(f"[INFO] 데이터 제한: {args.limit}건")

    # ----- 모델 & 토크나이저 -----
    model, tokenizer = load_model_and_tokenizer(model_name, dtype_str)

    # Gradient checkpointing (VRAM 절약)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # LoRA 적용
    model = apply_lora(model, lora_cfg)

    # ----- 토크나이즈 -----
    max_length = train_cfg["max_length"]
    print(f"[INFO] 토크나이즈 (max_length={max_length})")

    tokenized = dataset.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, max_length),
        remove_columns=dataset["train"].column_names,
        num_proc=4,
        desc="Tokenizing",
    )

    print(f"  train: {len(tokenized['train']):,}건")
    if "val" in tokenized:
        print(f"  val: {len(tokenized['val']):,}건")

    # ----- Data Collator -----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ----- Training Arguments -----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["lr"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=True,
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_strategy="steps" if "val" in tokenized else "no",
        eval_steps=train_cfg["eval_steps"] if "val" in tokenized else None,
        save_total_limit=3,
        load_best_model_at_end="val" in tokenized,
        metric_for_best_model="eval_loss" if "val" in tokenized else None,
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("val"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ----- 학습 -----
    print("\n[INFO] 학습 시작")
    resume_from = None
    if args.resume:
        # 가장 최근 체크포인트 탐색
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"[INFO] 체크포인트에서 재개: {resume_from}")
        else:
            print("[WARN] 체크포인트를 찾을 수 없어 처음부터 시작합니다.")

    trainer.train(resume_from_checkpoint=resume_from)

    # ----- 저장 -----
    final_dir = output_dir / "final"
    print(f"\n[INFO] 최종 LoRA 어댑터 저장: {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("[DONE] 학습 완료")


if __name__ == "__main__":
    main()
