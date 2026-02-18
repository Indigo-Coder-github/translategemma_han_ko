# 한문 번역기 (Hanja Translator)

조선시대 한문(漢文) → 현대 한국어 번역 모델을 구축하는 프로젝트.
TranslateGemma를 base로 한문-한국어 병렬 코퍼스에 파인튜닝한다.

실록에서 시작하되, 향후 문집·상소문·일기 등 다양한 조선시대 한문 장르로 확장 가능한 구조를 갖춘다.

## 프로젝트 구조

```
hanja-translator/
├── CLAUDE.md
├── data/
│   ├── raw/                    # 원본 데이터 (XML 등)
│   │   └── sillok/             # 조선왕조실록
│   ├── parsed/                 # 파싱된 jsonl (소스별 하위 디렉토리)
│   ├── processed/              # 전처리 중간 결과물 (소스별 하위 디렉토리)
│   ├── splits/                 # train / val / test (+ hf_dataset/)
│   └── eval/                   # 평가 결과
├── scripts/
│   ├── parsers/                # 소스별 파서 (확장 가능)
│   │   └── parse_sillok.py
│   ├── scrape_sillok_korean.py # 국역 수집
│   ├── prepare_pairs.py        # 필터링, 중복 제거, 역자 주석, variant 생성
│   ├── align_and_chunk.py      # 문장 정렬 + sliding window 청킹
│   ├── build_dataset.py        # 분할 + instruction 포맷 + HF Dataset
│   ├── validate_pairs.py       # 데이터 정합성 검증
│   └── evaluate.py             # 통합 평가 (instruction + baseline)
├── training/
│   ├── finetune_lora.py        # LoRA 파인튜닝
│   └── configs/                # 학습 하이퍼파라미터
├── inference/
│   └── translate.py            # 추론 (HF / vLLM)
├── demo/
│   └── app.py                  # Gradio 데모 UI (미구현)
└── requirements.txt
```

## 환경

- **연구용 서버:** L40s × 2 (파인튜닝 및 실험)
- **배포용 서버:** L40s × 2 (데모 서빙)
- **로컬 PC:** RTX 3060Ti (간단한 테스트, 4b 모델 추론)
- **Python:** 3.10+, 필요한 라이브러리 대부분 설치됨
- **주요 라이브러리:** transformers, peft, accelerate, datasets, gradio

## 베이스 모델

Google TranslateGemma (2026년 1월 공개, Gemma 3 기반 번역 특화 모델)

| 모델 | HuggingFace | 비고 |
|---|---|---|
| 4B | `google/translategemma-4b-it` | 로컬 테스트용 |
| 12B | `google/translategemma-12b-it` | 파인튜닝 1순위 후보 |
| 27B | `google/translategemma-27b-it` | L40s 2장 LoRA 가능 |

라이선스: Gemma License

### Gemma 3 / TranslateGemma 주의사항

- **bf16 필수**: fp16 → NaN logits → pad token만 출력 (치명적)
- **Windows bitsandbytes**: 4bit/8bit 양자화 시 pad token만 출력 (미작동)
- **모델 클래스**: `AutoModelForCausalLM` 사용 (`AutoModelForImageTextToText` 아님)
- **`dtype` 파라미터**: `torch_dtype`은 deprecated, `dtype` 사용
- **chat template**: source_lang_code/target_lang_code로 영어 system prompt 자동 생성
- **vLLM**: 원본 google/translategemma-* 로드 불가 (rope_parameters 검증 오류)
  - PR #32819 미머지 (2026.2 기준)
  - 워크어라운드: `Infomaniak-AI/vllm-translategemma-{4b,12b,27b}-it` 사용
  - delimiter 포맷: `<<<source>>>zh<<<target>>>ko<<<text>>>한문`

### VRAM / 모델 설정

| 설정 | 모델 | 방법 | VRAM | GPU |
|---|---|---|---|---|
| 추천 | 12B | bf16 LoRA, seq=2048 | ~30GB | 1x L40s |
| 대안 1 | 12B | bf16 LoRA, seq=4096 | ~36GB | 1x L40s |
| 대안 2 | 27B | QLoRA 4bit, seq=2048 | ~22GB | 1x L40s (Unsloth) |
| 대안 3 | 27B | bf16 LoRA, seq=2048 | ~62GB | 2x L40s |

## 데이터 소스

### Phase 1: 조선왕조실록 ✅ 채택

공공데이터포털에서 XML 파일 다운로드. 이용허락 제한 없음, 무료.

- **실록원문 (태조~철종):** https://www.data.go.kr/data/15053647/fileData.do
- **고순종실록 원문:** https://www.data.go.kr/data/15053646/fileData.do
- **부가정보 인물 데이터:** https://www.data.go.kr/data/15053645/fileData.do

형식: XML (원문만 포함, 국역 없음), 인코딩: UTF-8
스키마: 한국사데이터베이스 DTD v1.3

국역(한국어 번역)은 sillok.history.go.kr JSON API로 별도 수집:
- `GET /search/collectView.do?id={day_id}` (일자 단위 배치)
- 총서(2-part ID)는 API 미지원, 본문(3-part ID)만 가능
- 응답의 `contentHg` (HTML) 우선 사용 (`content`는 각주 인라인 혼입 문제)
- `footnoteHg`에서 각주를 별도 추출하여 `footnotes` 필드로 분리 저장
- 저작권: 학술·연구 목적 이용 가능, 상업적 이용 불가 (국사편찬위원회)

**API 필드 상세** (`sillokResult[]` 내 국역 항목):

| 필드 | 내용 | 사용 여부 |
|------|------|----------|
| `content` | plain text, 각주가 본문에 인라인 혼입 | ❌ fallback만 |
| `contentHg` | HTML, 각주는 `<sup>` 마커로만 표시 | ✅ **우선 사용** |
| `footnoteHg` | HTML, 각주 목록 (`[註 001]`, `[註 002]` 등) | ✅ 별도 저장 |

`contentHg`에서 `<sup>` 태그 제거 → `<a class="footnote_super">` 앵커+trailing space 제거 → HTML 태그 제거 → 엔티티 디코딩으로 깨끗한 번역문 추출.

XML 파일 규칙:
- 왕별 파일 분리, 파일명 가운데 영문자 = 왕조 순서 (a=태조, d=세종)
- 끝 숫자 = 재위년도
- `_000.xml` = 총서, `_200~` = 별첨/부록
- 고유명사에 REF 속성으로 ID 부여

### Phase 2 이후: 확장 후보 (미착수)

- **한국고전종합DB (db.itkc.or.kr):** 문집, 상소문 등 장르 다양. 단, Open API는 메타데이터 수준이고 크롤링은 저작권 문제 있음. 별도 데이터 제공 요청 또는 공모전 활용 등 검토 필요.
- **승정원일기:** 국역 미완성. 기계번역 결과만 있어 학습 데이터로 부적합.

### 장기 방향: 멀티모달 확장 (참고)

TranslateGemma는 Gemma 3 기반 멀티모달 모델로 I2T(이미지→번역) 지원. 단, 고문서 OCR은 중국산 전용 모델(예: PaddleOCR 등)이 더 효율적이므로, 현실적 파이프라인은:
- **OCR 모델(외부)로 한문 판독 → T2T 번역 모델로 번역** (2단계)
- 판독 결과를 사람이 검증 가능하여 학술적 신뢰도 높음
- Gemma 3는 IT2T(이미지+텍스트→번역)도 아키텍처상 가능하나, 학습 데이터 구축 비용 고려 필요

## 데이터 현황

- **원문:** 414,024건, 7,259만자 (XML 파싱 완료 → `data/parsed/sillok/articles.jsonl`)
- **국역:** 태조부터 수집 진행중 (~158,000 일자 API 요청, `--resume`으로 재개 가능)
- **국역/원문 비율:** 평균 2.3x → 병렬 코퍼스 약 1.7억자 규모 추정

### 국역 수집 진행 상황

| 왕대 | king_code | 상태 |
|------|-----------|------|
| 태조 ~ 성종 | aa ~ ia | ✅ 완료 |
| 연산군 ~ 명종 | ja ~ ma | ✅ 완료 |
| 선조 ~ 현종개수 | na ~ rb | ✅ 완료 |
| 숙종 ~ 철종 | sa ~ ya | 🔄 수집 중 |
| 고종, 순종 (제외) | za ~ zc | ⬜ 미수집 |

### 데이터 특성

- 왕대별 편차 큼: 중종 829만자 ~ 정종 9만자
- 조선 후기(순조, 헌종, 철종) 기록 간소화 경향
- 이본/수정본(선조수정, 숙종보궐정오 등)은 본편보다 기사가 길고 충실
- 짧은 기사(10자 미만) 38,870건(9.4%): 대부분 정형화된 반복 표현 (경연, 천문 등)
  - "○御夕講。" 3,150건, "○御晝講。" 3,069건 등 대량 중복
- 국역의 75%에 괄호 한자 주석 포함: `홍언필(洪彦弼)`
- XML `<index>` 태그에 고유명사 정보 (이름/지명/관직/서명/연호)

## 데이터 파이프라인

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
  │   랜덤 80:10:10 분할 → Gemma 3 instruction 포맷 → HF Dataset 저장
  │
  data/splits/{train,val,test}.jsonl + data/splits/hf_dataset/
```

### Step 1: prepare_pairs.py

- **필터링**: `translation`이 null/빈 문자열인 레코드 제거
- **중복 제거**: `(original, translation)` MD5 해시 기준 첫 번째만 유지
- **trailing space 후처리**: `(漢字) 조사` → `(漢字)조사` (기존 데이터 재수집 없이 적용)
- **역자 주석 제거** (`--note-detection strict|relaxed|off`):
  - `term(漢字) 짧은설명.` 패턴에서 설명 부분만 제거
  - strict 모드 탐지 기준:
    - ≤10자 교차참조 (예: `고려.`, `태종(太宗).`)
    - 11~20자 명사구 (서술형 어미 없는 경우, 예: `왕이 거처하던 궁전.`)
    - 알려진 종결 패턴 (`~을 가리킴.`, `~의 파자.`, `~을 의미한 것임.` 등)
  - orphaned term 탐지: `term(漢字)`가 텍스트 앞부분에 이미 등장하면 중복 각주로 판단, term까지 함께 제거
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

### Step 1-1: validate_pairs.py (검증)

- 한자 불일치: 번역의 `(漢字)`가 원문에 미포함
- 중복 용어: 동일 `term(漢字)`가 번역에 2회+ 등장 (각주 인라인 삽입 의심)
- 인라인 각주: `(漢字) 짧은설명.` 패턴 잔존
- trailing space: `(漢字) 조사` 패턴
- `--dump-flagged`로 플래그된 기사 목록 JSONL 저장

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

- **분할**: 랜덤 셔플 80:10:10 (seed=42, 재현 가능)
  - 제외: za~zc (고종/순종) — 근대 문체 전환기, 일본어·서양 용어 혼용
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

Packing은 학습 시점에 DataCollator에서 attention mask 분리로 구현 예정.

### 파이프라인 실행 결과 (2026-02-18, 태조~현종개수)

**Step 1: prepare_pairs.py**

| 항목 | 수치 |
|------|------|
| 입력 (articles_with_korean.jsonl) | 287,975건 |
| 번역 있음 | 287,085건 |
| 번역 없음 (제거) | 890건 |
| 중복 제거 | 26,392건 |
| **순수 출력** | **260,693건** |
| 역자 주석 제거 | 24,446건 (strict 모드) |
| variant: clean | 143,755 (55.1%) |
| variant: annotated | 70,472 (27.0%) |
| variant: mixed | 46,466 (17.8%) |

> 설정 비율 50/30/20 대비 clean이 약간 높은 이유: 괄호 한자가 없는 기사는 무조건 clean으로 분류되기 때문.

**데이터 검증: validate_pairs.py**

| 검증 항목 | 건수 | 비고 |
|-----------|------|------|
| trailing space | 52 (0.02%) | 거의 완전 해결 |
| 한자 불일치 | 7,496 | 번역의 `(漢字)`가 원문에 미포함 |
| 중복 용어 | 60,773 | 동일 인물/용어 반복 언급 (대부분 정상) |
| 인라인 각주 | 15,022 | `content` 폴백 기사에서 주로 발생 |
| 총 플래그 기사 | 27,865 (10.7%) | |

**Step 2: align_and_chunk.py**

| 항목 | 수치 |
|------|------|
| 그대로 통과 (≤2048 tokens) | 251,606건 (96.5%) |
| 청킹 대상 (>2048 tokens) | 9,081건 |
| 생성된 청크 수 | 102,879개 (평균 11.3청크/기사) |
| 분할 불가 (초과 길이로 통과) | 6건 |
| **총 출력** | **354,491건** |

**Step 3: build_dataset.py (랜덤 80:10:10, seed=42)**

| Split | 건수 | 비율 | 평균 tokens | clean | annotated | mixed |
|-------|------|------|------------|-------|-----------|-------|
| train | 283,592 | 80.0% | 412 | 53.5% | 28.2% | 18.3% |
| val | 35,449 | 10.0% | 414 | 54.0% | 28.2% | 17.8% |
| test | 35,450 | 10.0% | 414 | 53.6% | 28.1% | 18.3% |

> 수집 미완료(숙종~철종) 데이터 추가 시 재실행 필요.

## 훈련 데이터 전략

### 시퀀스 길이
- **seq=2048 tokens** (TranslateGemma 사전학습 컨텍스트와 동일)
- 전체 기사의 85%가 2K 이내에 수용됨

### 짧은 기사 처리
- **Packing** (attention mask 분리): 여러 짧은 기사를 2K에 채움
- **Deduplicate**: 동일 원문-번역 쌍 중복 제거
  - "○御夕講。→석강에 나아가다" 같은 반복 3,150건 → 1건

### 긴 기사 처리 (2K 초과)
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

## 학습

### LoRA 파인튜닝 스크립트 ✅

- `training/finetune_lora.py` + `training/configs/default.yaml`
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

### Baseline 평가 결과 (TranslateGemma 4B zero-shot)

- `scripts/evaluate.py --baseline`
- zh vs ja source_lang 비교: 차이 미미 → **zh 채택**
- BLEU < 5, chrF < 15 → 파인튜닝 필수 확인
- 문체 불일치: 모델은 "합니다체", 국역은 "하였다체" → BLEU 저하 원인
- 고유명사 오류: 배극렴→배클렴 등 조선 인명/관직명 미인식
- 결과: `data/eval/baseline_results.jsonl`

### 추론 스크립트 ✅

- `inference/translate.py` — HF transformers / vLLM 엔진 선택 가능
- vLLM: 현재 NotImplementedError (PR #32819 미머지)

## 관련 연구

- **H2KE (Son et al., EMNLP 2022)**: 동일 실록 데이터, mBART 기반, 한문→구역→현대한국어 2단계
- **Khayrallah & Koehn (2018)**: 타겟에 소스 언어 복사가 NMT에 가장 치명적
- **Don't Just Scratch the Surface (IJCNLP 2019)**: 한자 주석이 의미 구분에 도움

## 주의사항

- 고전 한문 ≠ 현대 중국어 (문법·어휘 상이, 조선 특유 용어 다수)
- TranslateGemma의 고전 한문 지원 범위 확인 필요
- 국역 번역은 1968~1993년 작업물이라 일부 표현이 현대 기준으로 어색할 수 있음
- 향후 데이터 소스 추가 시 도메인(장르) 간 문체 차이 고려

## TODO

### P2: 학습 (데이터 준비 완료 후)

- [ ] **L40s 서버에서 파인튜닝 실행** — `training/finetune_lora.py`
  - 1차 타겟: 12B bf16 LoRA, seq=2048, 1x L40s (~30GB)
  - packing (attention mask 분리) 구현 — 짧은 기사 효율적 처리
  - wandb 로깅 연동
- [ ] **평가 스크립트 확장**
  - fine-tuned 모델 로드 지원
  - 지표 추가: COMET, 도메인 용어(관직명/제도) 정확도
  - 왕대별 성능 breakdown
- [ ] **Gradio 데모** — `demo/app.py`, 한자 병기 여부 선택 (clean/annotated/mixed)

### P3: 확장 (핵심 완료 후)

- [ ] **XML `<index>` 태그 활용** — 인명/관직/지명 고유명사 사전 구축
- [ ] **데이터 소스 확장** — 한국고전종합DB (문집, 상소문, 일기 등)

## 코드 스타일

- Python 3.10+
- 타입 힌트 사용
- docstring 작성
- 경로: pathlib
- 설정: argparse 또는 yaml config
