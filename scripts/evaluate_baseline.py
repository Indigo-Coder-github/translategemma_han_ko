"""Baseline 평가: TranslateGemma zero-shot 한문 → 한국어 번역 테스트.

TranslateGemma의 전용 chat template(source_lang_code/target_lang_code)을
사용하여 조선시대 한문 원문을 번역하고, 국역(reference)이 있으면
BLEU/chrF를 계산한다.

Usage:
    # source_lang_code 비교 (zh vs ja)
    python scripts/evaluate_baseline.py --source-langs zh,ja --limit 5

    # 단일 source_lang으로 평가
    python scripts/evaluate_baseline.py --source-lang zh --limit 20

    # 서버 12B 평가 (bf16이 기본이므로 --quantize 불필요)
    python scripts/evaluate_baseline.py --model google/translategemma-12b-it
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_model_and_processor(
    model_id: str,
    quantize: str,
) -> tuple:
    """TranslateGemma 모델과 프로세서를 로드한다.

    NOTE: Gemma 3 계열은 반드시 bf16을 사용해야 한다.
    fp16은 수치 오버플로우로 NaN이 발생한다.
    bitsandbytes 양자화는 Windows에서 정상 동작하지 않을 수 있다.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)

    # Gemma 3는 bf16 필수 (fp16 → NaN logits)
    load_kwargs: dict = {"device_map": "auto", "dtype": torch.bfloat16}

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

    print(f"[INFO] 모델 로드 중: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    print(f"[INFO] 모델 로드 완료")

    return model, processor


# ---------------------------------------------------------------------------
# 번역
# ---------------------------------------------------------------------------

def translate(
    model,
    processor,
    text: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
) -> tuple[str, int, float]:
    """TranslateGemma로 번역하고 (번역문, 생성토큰수, 소요시간)을 반환."""
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
        }],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.time() - start

    gen_ids = output_ids[0][input_len:]
    gen_tokens = len(gen_ids)
    decoded = processor.decode(gen_ids, skip_special_tokens=True)

    return decoded.strip(), gen_tokens, elapsed


# ---------------------------------------------------------------------------
# 평가 지표
# ---------------------------------------------------------------------------

def compute_metrics(hypothesis: str, reference: str) -> dict[str, float]:
    """BLEU, chrF를 계산한다."""
    import sacrebleu

    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    chrf = sacrebleu.sentence_chrf(hypothesis, [reference])

    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
    }


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_data(
    input_path: Path,
    min_length: int,
    limit: int,
) -> list[dict]:
    """JSONL에서 translation이 있는 레코드를 로드한다."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # translation이 없으면 스킵
            if not record.get("translation"):
                continue
            # 원문 최소 길이 필터 (정형 반복문 제외)
            if len(record.get("original", "")) < min_length:
                continue
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break

    return records


# ---------------------------------------------------------------------------
# 콘솔 출력
# ---------------------------------------------------------------------------

def print_result(idx: int, record: dict, source_lang: str) -> None:
    """번역 결과를 콘솔에 출력한다."""
    print(f"\n{'='*70}")
    print(f"[{idx+1}] {record['article_id']}  (source_lang={source_lang})")
    print(f"{'='*70}")

    orig = record["original"]
    print(f"  원문:      {orig[:100]}{'...' if len(orig) > 100 else ''}")

    gen = record["generated"]
    print(f"  모델 출력: {gen[:200]}{'...' if len(gen) > 200 else ''}")

    ref = record.get("reference", "")
    if ref:
        print(f"  정답 국역: {ref[:200]}{'...' if len(ref) > 200 else ''}")

    if "bleu" in record:
        print(f"  BLEU={record['bleu']:.1f}  chrF={record['chrf']:.1f}  "
              f"tokens={record['gen_tokens']}  time={record['gen_time_sec']:.1f}s")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TranslateGemma baseline 평가 (한문 → 한국어)")

    parser.add_argument("--input", type=Path,
                        default=Path("data/parsed/sillok/articles_with_korean.jsonl"),
                        help="입력 JSONL 파일")
    parser.add_argument("--output", type=Path,
                        default=Path("data/eval/baseline_results.jsonl"),
                        help="결과 저장 경로")
    parser.add_argument("--model", type=str,
                        default="google/translategemma-4b-it",
                        help="모델 ID")
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "8bit", "4bit"],
                        help="양자화 방식 (기본: none=bf16, Windows에서 4bit/8bit 미작동 가능)")
    parser.add_argument("--source-lang", type=str, default="zh",
                        help="단일 source_lang_code (기본: zh)")
    parser.add_argument("--source-langs", type=str, default=None,
                        help="복수 source_lang_code 비교 (쉼표 구분, 예: zh,ja)")
    parser.add_argument("--target-lang", type=str, default="ko",
                        help="target_lang_code (기본: ko)")
    parser.add_argument("--limit", type=int, default=0,
                        help="최대 처리 건수 (0=전체)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="최대 생성 토큰 수")
    parser.add_argument("--min-length", type=int, default=10,
                        help="원문 최소 글자 수 (정형문 필터링)")

    args = parser.parse_args()

    # source_lang 결정
    if args.source_langs:
        source_langs = [s.strip() for s in args.source_langs.split(",")]
    else:
        source_langs = [args.source_lang]

    print(f"[INFO] source_lang_code: {source_langs}")
    print(f"[INFO] target_lang_code: {args.target_lang}")

    # 데이터 로드
    records = load_data(args.input, args.min_length, args.limit)
    print(f"[INFO] 입력 데이터: {len(records)}건 (min_length={args.min_length})")

    if not records:
        print("[ERROR] 처리할 데이터가 없습니다.")
        return

    # 모델 로드
    model, processor = load_model_and_processor(args.model, args.quantize)

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
            generated, gen_tokens, gen_time = translate(
                model, processor, original,
                source_lang, args.target_lang, args.max_new_tokens,
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
                metrics = compute_metrics(generated, reference)
                result.update(metrics)

            lang_results.append(result)
            all_results.append(result)

            # 콘솔 출력
            print_result(i, result, source_lang)

            # VRAM 절약 (bf16 4B ≈ 8.6GB, RTX 3060Ti 8GB에서 빠듯)
            torch.cuda.empty_cache()

        # lang별 요약
        if lang_results:
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

    # 결과 저장
    with open(args.output, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n[DONE] 결과 저장: {args.output} ({len(all_results)}건)")


if __name__ == "__main__":
    main()
