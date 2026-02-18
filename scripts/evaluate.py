"""통합 평가 스크립트: TranslateGemma 한문 → 한국어 번역 평가.

기본 모드(instruction)와 baseline 모드(chat template)를 모두 지원한다.

Usage:
    # LoRA 모델 평가 — HF 엔진 (기본)
    python scripts/evaluate.py \
        --model google/translategemma-12b-it \
        --adapter output/translategemma-12b-it-lora/final \
        --input data/splits/test.jsonl --limit 50

    # base 모델 instruction 포맷 평가 (adapter 없이)
    python scripts/evaluate.py \
        --model google/translategemma-4b-it \
        --input data/splits/test.jsonl --limit 10

    # baseline 모드 (기존 chat template zero-shot)
    python scripts/evaluate.py --baseline \
        --model google/translategemma-4b-it \
        --input data/parsed/sillok/articles_with_korean.jsonl \
        --source-lang zh --limit 5

    # vLLM 엔진 (미래)
    python scripts/evaluate.py --engine vllm \
        --model google/translategemma-12b-it \
        --adapter output/translategemma-12b-it-lora/final \
        --tensor-parallel 2 --input data/splits/test.jsonl
"""

from __future__ import annotations

import abc
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 평가 엔진 인터페이스
# ---------------------------------------------------------------------------

class EvalEngine(abc.ABC):
    """평가 엔진 추상 인터페이스."""

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        """chat messages에서 생성하고 (생성문, 토큰수, 소요시간)을 반환한다."""

    @abc.abstractmethod
    def generate_baseline(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        """baseline chat template으로 번역하고 (번역문, 토큰수, 소요시간)을 반환한다."""

    @abc.abstractmethod
    def close(self) -> None:
        """리소스 정리."""


# ---------------------------------------------------------------------------
# HuggingFace transformers 엔진
# ---------------------------------------------------------------------------

class HFEngine(EvalEngine):
    """HuggingFace transformers 기반 평가 엔진.

    NOTE: Gemma 3 계열은 bf16 필수. fp16 → NaN logits.
    bitsandbytes 양자화는 Windows에서 미작동할 수 있다.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        quantize: str = "none",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        # baseline용 프로세서
        self.processor = AutoProcessor.from_pretrained(model_id)
        # instruction용 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        load_kwargs: dict = {
            "device_map": "auto",
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

        # LoRA 어댑터 로드
        if adapter_path:
            from peft import PeftModel
            print(f"[INFO] LoRA 어댑터 로드 중: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
            print("[INFO] LoRA merge_and_unload 완료")

        print("[INFO] HF 모델 로드 완료")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        inputs = self.tokenizer.apply_chat_template(
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
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        torch.cuda.empty_cache()
        return decoded.strip(), gen_tokens, elapsed

    def generate_baseline(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> tuple[str, int, float]:
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": source_lang,
                "target_lang_code": target_lang,
                "text": text,
            }],
        }]

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

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# vLLM 엔진
# ---------------------------------------------------------------------------

class VLLMEngine(EvalEngine):
    """vLLM 기반 평가 엔진 (현재 미지원).

    TranslateGemma는 nested rope_parameters 포맷을 사용하는데,
    vLLM이 이를 파싱하지 못해 로드에 실패한다.
    (vllm-project/vllm PR #32819 미머지, 2026.2 기준)
    """

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "vLLM 엔진은 현재 TranslateGemma를 지원하지 않습니다.\n"
            "원인: nested rope_parameters 검증 오류 "
            "(vllm-project/vllm PR #32819 미머지, 2026.2 기준)\n"
            "--engine hf 를 사용하세요."
        )

    def generate(self, messages, max_new_tokens):
        raise NotImplementedError

    def generate_baseline(self, text, source_lang, target_lang, max_new_tokens):
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------------------------------------------------------
# 엔진 팩토리
# ---------------------------------------------------------------------------

def create_engine(args: argparse.Namespace) -> EvalEngine:
    """CLI 인자에 따라 적절한 엔진을 생성한다."""
    if args.engine == "vllm":
        return VLLMEngine(
            model_id=args.model,
            adapter_path=args.adapter,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        return HFEngine(
            model_id=args.model,
            adapter_path=args.adapter if not args.baseline else None,
            quantize=args.quantize,
        )


# ---------------------------------------------------------------------------
# 메트릭
# ---------------------------------------------------------------------------

def compute_sentence_metrics(hypothesis: str, reference: str) -> dict[str, float]:
    """sentence BLEU, chrF를 계산한다."""
    import sacrebleu

    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    chrf = sacrebleu.sentence_chrf(hypothesis, [reference])

    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
    }


def compute_corpus_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """corpus BLEU, chrF를 계산한다."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    return {
        "corpus_bleu": round(bleu.score, 2),
        "corpus_chrf": round(chrf.score, 2),
    }


def compute_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
) -> float | None:
    """COMET 점수를 계산한다. 미설치 시 None 반환."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("[WARN] COMET 미설치 (pip install unbabel-comet). COMET 평가 스킵.")
        return None

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=8, gpus=1)
    return round(output.system_score, 4)


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_baseline_data(
    input_path: Path,
    min_length: int,
    limit: int,
) -> list[dict]:
    """baseline 모드: translation이 있는 레코드를 로드한다."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record.get("translation"):
                continue
            if len(record.get("original", "")) < min_length:
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    return records


def load_instruction_data(
    input_path: Path,
    limit: int,
) -> list[dict]:
    """instruction 모드: instruction + input + output 필드가 있는 레코드를 로드한다."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record.get("instruction") or not record.get("input") or not record.get("output"):
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    return records


# ---------------------------------------------------------------------------
# 메시지 구성
# ---------------------------------------------------------------------------

def build_instruction_messages(record: dict) -> list[dict]:
    """레코드의 instruction/input/context 필드에서 chat messages를 구성한다.

    build_dataset.py의 format_prompt()와 동일한 유저 프롬프트를 재구성하되,
    tokenizer.apply_chat_template()에 전달할 수 있는 messages 형태로 반환한다.
    """
    parts: list[str] = [record["instruction"]]

    ctx_in = record.get("context_input")
    ctx_out = record.get("context_output")
    if ctx_in and ctx_out:
        parts.append("")
        parts.append("[맥락 - 이전 문장]")
        parts.append(f"원문: {ctx_in}")
        parts.append(f"번역: {ctx_out}")
        parts.append("")
        parts.append("[번역할 문장]")

    parts.append("")
    parts.append(record["input"])

    user_content = "\n".join(parts)
    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Baseline 모드
# ---------------------------------------------------------------------------

def run_baseline(engine: EvalEngine, args: argparse.Namespace) -> None:
    """기존 baseline 평가 로직 (TranslateGemma chat template zero-shot)."""
    # source_lang 결정
    if args.source_langs:
        source_langs = [s.strip() for s in args.source_langs.split(",")]
    else:
        source_langs = [args.source_lang]

    print(f"[INFO] baseline 모드")
    print(f"[INFO] source_lang_code: {source_langs}")
    print(f"[INFO] target_lang_code: {args.target_lang}")

    # 데이터 로드
    records = load_baseline_data(args.input, args.min_length, args.limit)
    print(f"[INFO] 입력 데이터: {len(records)}건 (min_length={args.min_length})")

    if not records:
        print("[ERROR] 처리할 데이터가 없습니다.")
        return

    # 출력 디렉토리
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for source_lang in source_langs:
        print(f"\n{'#'*70}")
        print(f"# source_lang_code = {source_lang}")
        print(f"{'#'*70}")

        lang_results = []

        for i, record in enumerate(records):
            original = record["original"]
            reference = record.get("translation", "")

            # 번역
            generated, gen_tokens, gen_time = engine.generate_baseline(
                original, source_lang, args.target_lang, args.max_new_tokens,
            )

            # 결과 구성
            result = {
                "article_id": record["article_id"],
                "original": original,
                "reference": reference,
                "source_lang": source_lang,
                "generated": generated,
                "gen_tokens": gen_tokens,
                "gen_time_sec": round(gen_time, 2),
            }

            # 지표 계산
            if reference:
                metrics = compute_sentence_metrics(generated, reference)
                result.update(metrics)

            lang_results.append(result)
            all_results.append(result)

            # 콘솔 출력
            _print_baseline_result(i, result, source_lang)

        # lang별 요약
        if lang_results:
            _print_baseline_summary(source_lang, lang_results)

    # 결과 저장
    with open(args.output, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n[DONE] 결과 저장: {args.output} ({len(all_results)}건)")


def _print_baseline_result(idx: int, result: dict, source_lang: str) -> None:
    """baseline 건별 결과를 콘솔에 출력한다."""
    print(f"\n{'='*70}")
    print(f"[{idx+1}] {result['article_id']}  (source_lang={source_lang})")
    print(f"{'='*70}")

    orig = result["original"]
    print(f"  원문:      {orig[:100]}{'...' if len(orig) > 100 else ''}")

    gen = result["generated"]
    print(f"  모델 출력: {gen[:200]}{'...' if len(gen) > 200 else ''}")

    ref = result.get("reference", "")
    if ref:
        print(f"  정답 국역: {ref[:200]}{'...' if len(ref) > 200 else ''}")

    if "bleu" in result:
        print(f"  BLEU={result['bleu']:.1f}  chrF={result['chrf']:.1f}  "
              f"tokens={result['gen_tokens']}  time={result['gen_time_sec']:.1f}s")


def _print_baseline_summary(source_lang: str, lang_results: list[dict]) -> None:
    """baseline lang별 요약을 출력한다."""
    refs_exist = [r for r in lang_results if "bleu" in r]
    if refs_exist:
        avg_bleu = sum(r["bleu"] for r in refs_exist) / len(refs_exist)
        avg_chrf = sum(r["chrf"] for r in refs_exist) / len(refs_exist)
    else:
        avg_bleu = avg_chrf = 0.0

    avg_time = sum(r["gen_time_sec"] for r in lang_results) / len(lang_results)
    avg_tokens = sum(r["gen_tokens"] for r in lang_results) / len(lang_results)

    print(f"\n[SUMMARY] source_lang={source_lang}")
    print(f"  건수: {len(lang_results)}")
    print(f"  평균 BLEU: {avg_bleu:.2f}")
    print(f"  평균 chrF: {avg_chrf:.2f}")
    print(f"  평균 생성 토큰: {avg_tokens:.0f}")
    print(f"  평균 소요 시간: {avg_time:.1f}s")


# ---------------------------------------------------------------------------
# Instruction 모드
# ---------------------------------------------------------------------------

def run_instruction(engine: EvalEngine, args: argparse.Namespace) -> None:
    """instruction 포맷 평가 (기본 모드). LoRA 어댑터 또는 base 모델."""
    print(f"[INFO] instruction 모드")
    if args.adapter:
        print(f"[INFO] LoRA 어댑터: {args.adapter}")
    else:
        print("[INFO] base 모델 (어댑터 없음)")

    # 데이터 로드
    records = load_instruction_data(args.input, args.limit)
    print(f"[INFO] 입력 데이터: {len(records)}건")

    if not records:
        print("[ERROR] 처리할 데이터가 없습니다.")
        return

    # 출력 디렉토리
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    hypotheses = []
    references_list = []
    sources = []

    for i, record in enumerate(records):
        # messages 구성 (instruction + input + context)
        messages = build_instruction_messages(record)
        reference = record["output"]

        # 생성
        generated, gen_tokens, gen_time = engine.generate(
            messages, args.max_new_tokens,
        )

        # sentence 메트릭
        metrics = compute_sentence_metrics(generated, reference)

        result = {
            "chunk_id": record.get("chunk_id", record.get("article_id", f"item_{i}")),
            "king_code": record.get("king_code", ""),
            "variant": record.get("variant", ""),
            "reference": reference,
            "generated": generated,
            "bleu": metrics["bleu"],
            "chrf": metrics["chrf"],
            "gen_tokens": gen_tokens,
            "gen_time_sec": round(gen_time, 2),
        }
        results.append(result)
        hypotheses.append(generated)
        references_list.append(reference)
        sources.append(record.get("input", record.get("original", "")))

        # 콘솔 건별 출력
        _print_instruction_result(i, result)

    # corpus-level 메트릭
    corpus = compute_corpus_metrics(hypotheses, references_list)

    # COMET (선택)
    comet_score = None
    if args.comet:
        comet_score = compute_comet(sources, hypotheses, references_list)
        if comet_score is not None:
            corpus["comet"] = comet_score

    # 콘솔 요약 출력
    _print_instruction_summary(results, corpus, comet_score)

    # 결과 JSONL 저장
    with open(args.output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # 마지막 줄에 요약 메타데이터
        summary = {
            "_summary": True,
            "n": len(results),
            **corpus,
        }
        if comet_score is not None:
            summary["comet"] = comet_score
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\n[DONE] 결과 저장: {args.output} ({len(results)}건)")


def _print_instruction_result(idx: int, result: dict) -> None:
    """instruction 건별 결과를 콘솔에 출력한다."""
    print(f"\n{'='*70}")
    print(f"[{idx+1}] {result['chunk_id']}  "
          f"variant={result['variant']}  king={result['king_code']}")
    print(f"{'='*70}")

    ref = result["reference"]
    print(f"  정답:      {ref[:200]}{'...' if len(ref) > 200 else ''}")

    gen = result["generated"]
    print(f"  모델 출력: {gen[:200]}{'...' if len(gen) > 200 else ''}")

    print(f"  BLEU={result['bleu']:.1f}  chrF={result['chrf']:.1f}  "
          f"tokens={result['gen_tokens']}  time={result['gen_time_sec']:.1f}s")


def _print_instruction_summary(
    results: list[dict],
    corpus: dict[str, float],
    comet_score: float | None,
) -> None:
    """instruction 모드 전체 요약을 출력한다."""
    print(f"\n{'#'*70}")
    print(f"# 전체 요약 (n={len(results)})")
    print(f"{'#'*70}")

    print(f"  corpus BLEU: {corpus['corpus_bleu']:.2f}")
    print(f"  corpus chrF: {corpus['corpus_chrf']:.2f}")
    if comet_score is not None:
        print(f"  COMET:       {comet_score:.4f}")

    # 평균 sentence 메트릭
    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    avg_chrf = sum(r["chrf"] for r in results) / len(results)
    avg_tokens = sum(r["gen_tokens"] for r in results) / len(results)
    avg_time = sum(r["gen_time_sec"] for r in results) / len(results)

    print(f"  avg sentence BLEU: {avg_bleu:.2f}")
    print(f"  avg sentence chrF: {avg_chrf:.2f}")
    print(f"  avg 생성 토큰: {avg_tokens:.0f}")
    print(f"  avg 소요 시간: {avg_time:.1f}s")

    # variant별 breakdown
    variant_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        v = r.get("variant", "unknown")
        variant_groups[v].append(r)

    if len(variant_groups) > 1:
        print(f"\n  --- variant별 breakdown ---")
        for variant in sorted(variant_groups.keys()):
            group = variant_groups[variant]
            v_bleu = sum(r["bleu"] for r in group) / len(group)
            v_chrf = sum(r["chrf"] for r in group) / len(group)
            print(f"  {variant:12s}: n={len(group):4d}  "
                  f"BLEU={v_bleu:.2f}  chrF={v_chrf:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TranslateGemma 한문→한국어 번역 평가 (instruction / baseline)")

    # 모드
    parser.add_argument("--baseline", action="store_true",
                        help="baseline 모드 (TranslateGemma chat template zero-shot)")

    # 엔진
    parser.add_argument("--engine", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="추론 엔진 (기본: hf)")

    # 모델
    parser.add_argument("--model", type=str,
                        default="google/translategemma-12b-it",
                        help="base 모델 ID (기본: google/translategemma-12b-it)")
    parser.add_argument("--adapter", type=str, default=None,
                        help="[instruction] LoRA 어댑터 경로")

    # 입출력
    parser.add_argument("--input", type=Path,
                        default=Path("data/splits/test.jsonl"),
                        help="입력 JSONL 파일")
    parser.add_argument("--output", type=Path,
                        default=Path("data/eval/eval_results.jsonl"),
                        help="결과 저장 경로 (기본: data/eval/eval_results.jsonl)")

    # 공통
    parser.add_argument("--limit", type=int, default=0,
                        help="최대 처리 건수 (0=전체)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="최대 생성 토큰 수")
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "8bit", "4bit"],
                        help="양자화 방식 (기본: none=bf16)")
    parser.add_argument("--min-length", type=int, default=10,
                        help="[baseline] 원문 최소 글자 수")

    # instruction 전용
    parser.add_argument("--comet", action="store_true",
                        help="[instruction] COMET 평가 활성화")

    # baseline 전용
    parser.add_argument("--source-lang", type=str, default="zh",
                        help="[baseline] source_lang_code (기본: zh)")
    parser.add_argument("--source-langs", type=str, default=None,
                        help="[baseline] 복수 source_lang_code 비교 (쉼표 구분)")
    parser.add_argument("--target-lang", type=str, default="ko",
                        help="[baseline] target_lang_code (기본: ko)")

    # vLLM 전용
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="[vLLM] tensor parallel GPU 수")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="[vLLM] GPU 메모리 사용 비율")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="[vLLM] 최대 시퀀스 길이")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"[INFO] 엔진: {args.engine}, 모델: {args.model}")

    engine = create_engine(args)

    try:
        if args.baseline:
            run_baseline(engine, args)
        else:
            run_instruction(engine, args)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
