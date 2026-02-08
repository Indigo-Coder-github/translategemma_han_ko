"""조선시대 한문 → 현대 한국어 번역 추론 스크립트.

TranslateGemma 모델을 사용하여 한문 원문을 한국어로 번역한다.
HuggingFace transformers와 vLLM 두 가지 엔진을 지원한다.

Usage:
    # HF 엔진 (기본, 로컬 테스트용)
    python inference/translate.py --engine hf --model google/translategemma-4b-it \
        --text "雨。 前此久旱, 及上卽位, 霈然下雨, 人心大悅。"

    # vLLM 엔진 (현재 미지원 — vLLM PR #32819 머지 대기중)
    # python inference/translate.py --engine vllm --model google/translategemma-12b-it \
    #     --tensor-parallel 2 --input data/test_input.jsonl --output data/test_output.jsonl

    # 파일 입력 (한 줄에 한문 한 건, 또는 JSONL의 "original" 필드)
    python inference/translate.py --engine hf --input input.txt --output output.jsonl

    # stdin 대화형
    echo "太祖卽位于壽昌宮。" | python inference/translate.py --engine hf
"""

from __future__ import annotations

import abc
import argparse
import json
import sys
import time
from pathlib import Path

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# TranslateGemma 메시지 포맷
# ---------------------------------------------------------------------------

def build_messages(
    text: str,
    source_lang: str = "zh",
    target_lang: str = "ko",
) -> list[dict]:
    """TranslateGemma chat template용 메시지를 구성한다."""
    return [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
        }],
    }]


# ---------------------------------------------------------------------------
# 엔진 인터페이스
# ---------------------------------------------------------------------------

class TranslationEngine(abc.ABC):
    """번역 엔진 추상 인터페이스."""

    @abc.abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        """번역하고 (번역문, 생성토큰수, 소요시간)을 반환한다."""

    @abc.abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> list[tuple[str, int, float]]:
        """배치 번역. (번역문, 생성토큰수, 소요시간) 리스트를 반환한다."""

    @abc.abstractmethod
    def close(self) -> None:
        """리소스 정리."""


# ---------------------------------------------------------------------------
# HuggingFace transformers 엔진
# ---------------------------------------------------------------------------

class HFEngine(TranslationEngine):
    """HuggingFace transformers 기반 추론 엔진.

    NOTE: Gemma 3 계열은 bf16 필수. fp16 → NaN logits.
    bitsandbytes 양자화는 Windows에서 미작동할 수 있다.
    """

    def __init__(
        self,
        model_id: str,
        quantize: str = "none",
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)

        load_kwargs: dict = {
            "device_map": device_map,
            "dtype": torch.bfloat16,
        }

        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("[INFO] 4bit 양자화 적용 (Windows에서 미작동 가능)")
        elif quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print("[INFO] 8bit 양자화 적용 (Windows에서 미작동 가능)")
        else:
            print("[INFO] bf16 전체 정밀도 로드")

        print(f"[INFO] HF 모델 로드 중: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        print("[INFO] HF 모델 로드 완료")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        torch = self.torch
        messages = build_messages(text, source_lang, target_lang)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        elapsed = time.time() - start

        gen_ids = output_ids[0][input_len:]
        gen_tokens = len(gen_ids)
        decoded = self.processor.decode(gen_ids, skip_special_tokens=True)

        torch.cuda.empty_cache()
        return decoded.strip(), gen_tokens, elapsed

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> list[tuple[str, int, float]]:
        # HF transformers는 동적 배칭이 없으므로 순차 처리
        return [
            self.translate(t, source_lang, target_lang, max_new_tokens)
            for t in texts
        ]

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# vLLM 엔진
# ---------------------------------------------------------------------------

class VLLMEngine(TranslationEngine):
    """vLLM 기반 추론 엔진 (현재 미지원).

    TranslateGemma는 nested rope_parameters 포맷을 사용하는데,
    vLLM이 이를 파싱하지 못해 로드에 실패한다.
    (vllm-project/vllm PR #32819 미머지, 2026.2 기준)

    PR 머지 후 이 클래스를 구현할 예정.
    """

    def __init__(self, **kwargs) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "vLLM 엔진은 현재 TranslateGemma를 지원하지 않습니다.\n"
            "원인: nested rope_parameters 검증 오류 "
            "(vllm-project/vllm PR #32819 미머지, 2026.2 기준)\n"
            "--engine hf 를 사용하세요."
        )

    def translate(self, text, source_lang, target_lang, max_new_tokens):
        raise NotImplementedError

    def translate_batch(self, texts, source_lang, target_lang, max_new_tokens):
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------------------------------------------------------
# 엔진 팩토리
# ---------------------------------------------------------------------------

def create_engine(args: argparse.Namespace) -> TranslationEngine:
    """CLI 인자에 따라 적절한 엔진을 생성한다."""
    if args.engine == "vllm":
        return VLLMEngine(
            model_id=args.model,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        return HFEngine(
            model_id=args.model,
            quantize=args.quantize,
        )


# ---------------------------------------------------------------------------
# 입력 로드
# ---------------------------------------------------------------------------

def load_inputs(input_path: Path | None, text: str | None) -> list[dict]:
    """입력을 로드한다. --text, --input, 또는 stdin에서 읽는다.

    반환: [{"original": "한문 원문", ...extra_fields}, ...]
    """
    records: list[dict] = []

    if text:
        records.append({"original": text})
        return records

    source = open(input_path, encoding="utf-8") if input_path else sys.stdin
    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            # JSONL 시도
            if line.startswith("{"):
                try:
                    record = json.loads(line)
                    if "original" in record:
                        records.append(record)
                        continue
                except json.JSONDecodeError:
                    pass
            # 일반 텍스트 (한 줄 = 한 건)
            records.append({"original": line})
    finally:
        if input_path and source is not sys.stdin:
            source.close()

    return records


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="한문 → 한국어 번역 (TranslateGemma)")

    # 엔진 선택
    parser.add_argument("--engine", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="추론 엔진 (기본: hf)")

    # 모델
    parser.add_argument("--model", type=str,
                        default="google/translategemma-4b-it",
                        help="모델 ID")

    # HF 전용 옵션
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "8bit", "4bit"],
                        help="[HF] 양자화 방식 (기본: none=bf16)")

    # vLLM 전용 옵션
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="[vLLM] tensor parallel GPU 수 (기본: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="[vLLM] GPU 메모리 사용 비율 (기본: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="[vLLM] 최대 시퀀스 길이 (기본: 모델 설정)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[vLLM] 배치 크기 (기본: 32)")

    # 번역 옵션
    parser.add_argument("--source-lang", type=str, default="zh",
                        help="source_lang_code (기본: zh)")
    parser.add_argument("--target-lang", type=str, default="ko",
                        help="target_lang_code (기본: ko)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="최대 생성 토큰 수 (기본: 2048)")

    # 입출력
    parser.add_argument("--text", type=str, default=None,
                        help="번역할 한문 텍스트 (단건)")
    parser.add_argument("--input", type=Path, default=None,
                        help="입력 파일 (텍스트 또는 JSONL, 미지정 시 stdin)")
    parser.add_argument("--output", type=Path, default=None,
                        help="출력 JSONL 파일 (미지정 시 stdout)")

    args = parser.parse_args()

    # 입력 로드
    records = load_inputs(args.input, args.text)
    if not records:
        print("[ERROR] 입력이 없습니다.", file=sys.stderr)
        return

    print(f"[INFO] 입력: {len(records)}건", file=sys.stderr)
    print(f"[INFO] 엔진: {args.engine}, 모델: {args.model}", file=sys.stderr)

    # 엔진 생성
    engine = create_engine(args)

    # 번역
    try:
        output_file = (
            open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        )

        if args.engine == "vllm" and len(records) > 1:
            # vLLM: 배치 처리
            for batch_start in range(0, len(records), args.batch_size):
                batch = records[batch_start:batch_start + args.batch_size]
                texts = [r["original"] for r in batch]

                results = engine.translate_batch(
                    texts, args.source_lang, args.target_lang,
                    args.max_new_tokens,
                )

                for record, (generated, gen_tokens, gen_time) in zip(batch, results):
                    out = {
                        "original": record["original"],
                        "generated": generated,
                        "gen_tokens": gen_tokens,
                        "gen_time_sec": round(gen_time, 2),
                    }
                    # 원본 레코드의 추가 필드 보존
                    for key in ("article_id", "king", "date", "title"):
                        if key in record:
                            out[key] = record[key]
                    output_file.write(
                        json.dumps(out, ensure_ascii=False) + "\n"
                    )

                done = min(batch_start + args.batch_size, len(records))
                print(
                    f"[INFO] {done}/{len(records)}건 완료",
                    file=sys.stderr,
                )
        else:
            # HF 또는 단건: 순차 처리
            for i, record in enumerate(records):
                generated, gen_tokens, gen_time = engine.translate(
                    record["original"], args.source_lang, args.target_lang,
                    args.max_new_tokens,
                )

                out = {
                    "original": record["original"],
                    "generated": generated,
                    "gen_tokens": gen_tokens,
                    "gen_time_sec": round(gen_time, 2),
                }
                for key in ("article_id", "king", "date", "title"):
                    if key in record:
                        out[key] = record[key]
                output_file.write(json.dumps(out, ensure_ascii=False) + "\n")

                if args.output:
                    output_file.flush()

                print(
                    f"[INFO] [{i+1}/{len(records)}] "
                    f"tokens={gen_tokens} time={gen_time:.1f}s "
                    f"| {generated[:60]}...",
                    file=sys.stderr,
                )

        if args.output:
            output_file.close()
            print(f"\n[DONE] 결과 저장: {args.output}", file=sys.stderr)

    finally:
        engine.close()


if __name__ == "__main__":
    main()
