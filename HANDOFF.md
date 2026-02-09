# 작업 인수인계 문서

> 다른 기기에서 작업 이어가기 위한 현재 상태 요약
> 작성일: 2026-02-07 (최종 갱신: 2026-02-10)

## 프로젝트 요약

조선시대 한문(漢文) → 현대 한국어 번역 모델 구축.
Google TranslateGemma (Gemma 3 기반)를 LoRA 파인튜닝.

## 현재 진행 상태

### 완료된 작업

1. **XML 파싱** ✅
   - `scripts/parsers/parse_sillok.py` → `data/parsed/sillok/articles.jsonl`
   - 743개 XML 파일 → 414,024건, 7,259만자

2. **국역 수집 스크립트** ✅ (2026-02-08 수정)
   - `scripts/scrape_sillok_korean.py`
   - 일자 단위 배치 API 요청 (158,860일)
   - sillok.history.go.kr JSON API, 랜덤 1~5초 대기
   - **수정**: `content` → `contentHg` 우선 사용 (각주 인라인 혼입 해결)
   - **수정**: `footnoteHg` 별도 파싱하여 `footnotes` 필드로 분리 저장

3. **데이터 분석** ✅
   - 왕대별 길이 분포, 기록 밀도 분석
   - 짧은 기사 패턴 파악 (38,870건, 9.4%)
   - 시각화: `data/parsed/sillok/king_original_analysis.png`, `injong_length_dist.png`

### 진행 중

4. **국역 재수집 실행 중** 🔄
   - 각주 인라인 혼입 문제 수정 후 전체 재수집 시작 (2026-02-08)
   - `python scripts/scrape_sillok_korean.py --delay-min 1 --delay-max 5`
   - 전체 약 4~5일 소요 예상
   - `--resume` 옵션으로 중단 시 재개 가능

5. **데이터 처리 파이프라인** ✅ (2026-02-08 완료)
   - 3단계 스크립트 구현 및 테스트 완료
   - 상세 내용은 아래 「데이터 처리 파이프라인 상세」 참조

6. **Baseline 평가** ✅ (2026-02-08 완료)
   - `scripts/evaluate_baseline.py`
   - TranslateGemma 4B zero-shot, 로컬 RTX 3060Ti (bf16)
   - zh vs ja source_lang 비교: 차이 미미 → **zh 채택**
   - BLEU < 5, chrF < 15 → 파인튜닝 필수 확인
   - 결과: `data/eval/baseline_results.jsonl`

7. **추론 스크립트** ✅ (2026-02-08 완료)
   - `inference/translate.py`
   - HF transformers / vLLM 엔진 선택 가능
   - vLLM: 현재 NotImplementedError (PR #32819 미머지)

8. **국역 수집기 각주 분리 수정** ✅ (2026-02-08 완료)
   - 문제: `content` 필드에 역자 각주가 본문에 인라인 혼입
     - 예: `시좌궁(時坐宮) 그 당시에 왕이 거처하던 궁전.` 이 본문에 삽입됨
   - 원인: API의 `content` 필드는 plain text로 각주를 본문에 포함
   - 해결: `contentHg` (HTML) 우선 사용, `<sup>` 각주 번호만 제거
   - 각주는 `footnoteHg` 필드에서 별도 추출하여 `footnotes` 필드로 저장
   - 검증: `waa_10107017_001` (태조 즉위) 기사에서 4개 각주 모두 분리 확인

9. **LoRA 파인튜닝 스크립트** ✅ (2026-02-10 완료)
   - `training/finetune_lora.py` — 메인 학습 스크립트
   - `training/configs/default.yaml` — 하이퍼파라미터 설정
   - HF Transformers `Trainer` + PEFT LoRA (rsLoRA)
   - 주요 설계:
     - Loss 마스킹: `<start_of_turn>model\n` 이후 토큰만 loss 계산
     - target modules: q/k/v/o_proj + gate/up/down_proj (7개)
     - rank=64, alpha=64, rsLoRA 활성화
     - gradient checkpointing + bf16 필수
     - 데이터: hf_dataset(arrow) 우선, 없으면 JSONL 자동 폴백
   - 실행: `python training/finetune_lora.py --config training/configs/default.yaml`
   - Multi-GPU: `accelerate launch training/finetune_lora.py ...` (FSDP2)
   - Resume: `--resume` 플래그로 마지막 체크포인트에서 재개
   - Smoke test: `--model google/translategemma-4b-it --limit 100`

### 미착수

10. **실제 파인튜닝 실행** (L40s 서버에서)
11. **Gradio 데모**

## 확정된 훈련 데이터 전략

### 시퀀스 길이
- **seq=2048 tokens** (TranslateGemma 사전학습 컨텍스트와 동일)
- 전체 기사의 85%가 2K 이내에 수용됨

### 짧은 기사 처리
- **Packing** (attention mask 분리): 여러 짧은 기사를 2K에 채움
- **Deduplicate**: 동일 원문-번역 쌍 중복 제거
  - "○御夕講。→석강에 나아가다" 같은 반복 3,150건 → 1건

### 긴 기사 처리 (2K 초과, 15%)
- **Sliding window**: 3~5문장씩 chunk 분할
- 이전 chunk 마지막 2~3문장을 context로 overlap (loss 미적용)
- 한문은 주어 생략이 많아 앞문맥 필수

### 괄호 한자 처리
- **Multi-variant + instruction conditioning**
  - Clean 50%: `홍언필이 아뢰기를` (instruction: "현대 한국어로 번역하라")
  - 한자 병기 30%: `홍언필(洪彦弼)이 아뢰기를` (instruction: "한자를 병기하여 번역하라")
  - 혼합 20%: 일부만 병기
- 역자 주석 (`도규(道揆) 재상을 가리킴.`)은 별도 제거

### 원문 소스
- XML 파싱 결과 사용 (API 원문은 교감주 `國(寶) 〔璽〕` 포함하므로 비채택)

### 데이터 분할 (왕대 기준)
- train: 태조 ~ 성종
- val: 연산군 ~ 명종
- test: 선조 ~ 철종

## VRAM / 모델 설정

| 설정 | 모델 | 방법 | VRAM | GPU |
|---|---|---|---|---|
| 추천 | 12B | bf16 LoRA, seq=2048 | ~30GB | 1x L40s |
| 대안 1 | 12B | bf16 LoRA, seq=4096 | ~36GB | 1x L40s |
| 대안 2 | 27B | QLoRA 4bit, seq=2048 | ~22GB | 1x L40s (Unsloth) |
| 대안 3 | 27B | bf16 LoRA, seq=2048 | ~62GB | 2x L40s |

## 관련 연구 (참고)

- **H2KE (Son et al., EMNLP 2022)**: 동일 실록 데이터, mBART 기반, 한문→구역→현대한국어 2단계
- **Khayrallah & Koehn (2018)**: 타겟에 소스 언어 복사가 NMT에 가장 치명적
- **Don't Just Scratch the Surface (IJCNLP 2019)**: 한자 주석이 의미 구분에 도움

## 파일 구조 (주요)

```
data/
  raw/sillok/                     # XML 원본 743개 + DTD + CSV
  parsed/sillok/
    articles.jsonl                # 원문 414,024건 (파싱 완료)
    articles_with_korean.jsonl    # 국역 수집 중 (현재 태조)
    articles_with_korean.jsonl.progress  # 수집 진행 상태
    king_original_analysis.png    # 왕대별 분석 차트
    injong_length_dist.png        # 인종 길이 분포 차트

scripts/
  parsers/parse_sillok.py         # XML 파서
  scrape_sillok_korean.py         # 국역 수집기
```

## 데이터 처리 파이프라인 상세

### 파이프라인 흐름

```
articles_with_korean.jsonl
  │
  ▼ [Step 1] python scripts/prepare_pairs.py
  │   필터링 → 중복 제거 → 역자 주석 제거 → 괄호 한자 variant 생성
  │
  data/processed/sillok/clean_pairs.jsonl
  │
  ▼ [Step 2] python scripts/align_and_chunk.py
  │   토큰 수 계산 → 2048 초과 기사 문장 분할 + sliding window 청킹
  │
  data/processed/sillok/chunked_pairs.jsonl
  │
  ▼ [Step 3] python scripts/build_dataset.py --save-hf
  │   왕대별 train/val/test 분할 → Gemma 3 instruction 포맷 → HF Dataset 저장
  │
  data/splits/{train,val,test}.jsonl + data/splits/hf_dataset/
```

### Step 1: prepare_pairs.py

- **필터링**: `translation`이 null/빈 문자열인 레코드 제거
- **중복 제거**: `(original, translation)` MD5 해시 기준 첫 번째만 유지
- **역자 주석 제거** (`--note-detection strict|relaxed|off`):
  - `term(漢字) 짧은설명.` 패턴에서 설명 부분만 제거
  - strict 모드 탐지 기준:
    - ≤10자 교차참조 (예: `고려.`, `태종(太宗).`)
    - 11~20자 명사구 (서술형 어미 없는 경우, 예: `왕이 거처하던 궁전.`)
    - 알려진 종결 패턴 (`~을 가리킴.`, `~의 파자.`, `~을 의미한 것임.` 등)
  - 주의: orphaned term 잔존 가능 (예: 주석 제거 후 `시좌궁` 단독 남음)
- **variant 생성** (기사당 1개, `hash(article_id)+seed` 기반 결정론적):
  - clean 50%: 모든 `(漢字)` 제거 → instruction "현대 한국어로 번역하라"
  - annotated 30%: `(漢字)` 유지 → instruction "한자를 병기하여 번역하라"
  - mixed 20%: 40%만 랜덤 유지 → instruction "필요한 부분에만 한자를 병기하여 번역하라"
  - 괄호 주석이 없는 기사는 항상 clean

```bash
python scripts/prepare_pairs.py \
  --input data/parsed/sillok/articles_with_korean.jsonl \
  --output data/processed/sillok/clean_pairs.jsonl \
  --seed 42 --clean-ratio 0.5 --annotated-ratio 0.3 --mixed-ratio 0.2
```

### Step 2: align_and_chunk.py

- **토크나이저**: `google/translategemma-4b-it` (없으면 자동 다운로드)
- **2048 토큰 이하**: 그대로 통과 (chunk_id = `{article_id}_c000`)
- **2048 토큰 초과** (~3~4%):
  - 원문: `。`(구점) 기준 분할
  - 국역: `. ! ?` 기준 분할
  - 문장 수 비슷하면(±30%) 위치 기반 1:1 정렬, 아니면 길이비 탐욕 정렬
  - 3~5문장쌍씩 청크, 각 청크가 2048 토큰 이내가 되도록 적응적 조절
  - overlap: 이전 청크의 마지막 2문장쌍을 context로 포함

```bash
python scripts/align_and_chunk.py \
  --model google/translategemma-4b-it \
  --max-tokens 2048 --chunk-size 4 --overlap 2
```

### Step 3: build_dataset.py

- **분할 기준** (king_code):
  - train: aa~ia (태조~성종)
  - val: ja~ma (연산군~명종)
  - test: na~ya (선조~철종)
  - 제외: za~zc (고종/순종)
- **instruction 포맷** (Gemma 3 턴 구조, 한국어 instruction):
  ```
  <bos><start_of_turn>user
  다음 조선시대 한문을 현대 한국어로 번역하라.

  {original}<end_of_turn>
  <start_of_turn>model
  {translation}<end_of_turn>
  ```
  - context가 있는 청크: `[맥락 - 이전 문장]` 섹션 추가
- **HF DatasetDict**: `--save-hf` 옵션으로 arrow 형식 저장

```bash
python scripts/build_dataset.py --save-hf
```

### Packing

학습 시점에 처리 (전처리 파이프라인 범위 밖).
HF DataCollator 또는 커스텀 collator에서 attention mask 분리로 구현 예정.

### 테스트 결과 (20건 샘플)

| 단계 | 입력 | 출력 | 비고 |
|------|------|------|------|
| Step 1 | 20건 | 20건 | 역자 주석 49건 제거, variant 분포 정상 |
| Step 2 | 20건 | 93건 | 16건 통과 + 4건 → 77청크 |
| Step 3 | 93건 | 93건 (train) | 전부 태조 기사, 평균 535 tokens |

## 파일 구조 (주요)

```
data/
  raw/sillok/                     # XML 원본 743개 + DTD + CSV
  parsed/sillok/
    articles.jsonl                # 원문 414,024건 (파싱 완료)
    articles_with_korean.jsonl    # 국역 수집 중 (현재 태조)
    articles_with_korean.jsonl.progress  # 수집 진행 상태
    king_original_analysis.png    # 왕대별 분석 차트
    injong_length_dist.png        # 인종 길이 분포 차트
  processed/sillok/               # 파이프라인 중간 결과물
    clean_pairs.jsonl             # Step 1 출력
    chunked_pairs.jsonl           # Step 2 출력
  splits/                         # 최종 학습 데이터
    train.jsonl
    val.jsonl
    test.jsonl
    hf_dataset/                   # HuggingFace arrow 형식

  eval/                          # 평가 결과
    baseline_results.jsonl        # Baseline 평가 (zh/ja × 3건)

scripts/
  parsers/parse_sillok.py         # XML 파서
  scrape_sillok_korean.py         # 국역 수집기
  prepare_pairs.py                # 파이프라인 Step 1
  align_and_chunk.py              # 파이프라인 Step 2
  build_dataset.py                # 파이프라인 Step 3
  evaluate_baseline.py            # Baseline 평가

training/
  finetune_lora.py                # LoRA 파인튜닝 (HF Trainer + PEFT)
  configs/default.yaml            # 하이퍼파라미터 설정

inference/
  translate.py                    # 추론 (HF / vLLM 선택)
```

## sillok.history.go.kr API 구조

`GET /search/collectView.do?id={day_id}` 응답의 `sillokResult[]` 내 국역(k 접두사) 항목 필드:

| 필드 | 내용 | 사용 여부 |
|------|------|----------|
| `content` | plain text, 각주가 본문에 인라인 혼입 | ❌ fallback만 |
| `contentHg` | HTML, 각주는 `<sup>` 마커로만 표시 | ✅ **우선 사용** |
| `footnoteHg` | HTML, 각주 목록 (`[註 001]`, `[註 002]` 등) | ✅ 별도 저장 |

`contentHg`에서 `<sup>` 태그 제거 → HTML 태그 제거 → 엔티티 디코딩으로 깨끗한 번역문 추출.

## Gemma 3 / TranslateGemma 주의사항

- **bf16 필수**: fp16 → NaN logits → pad token만 출력 (치명적)
- **Windows bitsandbytes**: 4bit/8bit 양자화 시 pad token만 출력 (미작동)
- **모델 클래스**: `AutoModelForCausalLM` 사용 (`AutoModelForImageTextToText` 아님)
- **`dtype` 파라미터**: `torch_dtype`은 deprecated, `dtype` 사용
- **vLLM**: 원본 google/translategemma-* 로드 불가 (rope_parameters 검증 오류)
  - PR #32819 미머지 (2026.2 기준)
  - 워크어라운드: `Infomaniak-AI/vllm-translategemma-{4b,12b,27b}-it` 사용
  - delimiter 포맷: `<<<source>>>zh<<<target>>>ko<<<text>>>한문`

## 다음 작업 제안 (우선순위)

1. **국역 재수집 완료 대기** — 현재 진행중, 완료 후 파이프라인 재실행
2. **L40s 서버에서 파인튜닝 실행** — `training/finetune_lora.py` 실행 + 결과 확인
3. **평가 스크립트 구현** — 학습 후 BLEU/chrF/COMET 측정
4. **XML `<index>` 태그 추출** — 고유명사 사전 구축
5. **Gradio 데모** — demo/ 구현
