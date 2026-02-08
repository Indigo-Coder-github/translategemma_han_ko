# 한문 번역기 (Hanja Translator)

조선시대 한문(漢文) → 현대 한국어 번역 모델을 구축하는 프로젝트.
TranslateGemma를 base로 한문-한국어 병렬 코퍼스에 파인튜닝한다.

실록에서 시작하되, 향후 문집·상소문·일기 등 다양한 조선시대 한문 장르로 확장 가능한 구조를 갖춘다.

## 프로젝트 구조

```
hanja-translator/
├── CLAUDE.md
├── HANDOFF.md                  # 작업 인수인계 문서 (현재 상태 요약)
├── data/
│   ├── raw/                    # 원본 데이터 (XML 등)
│   │   ├── sillok/             # 조선왕조실록
│   │   └── (향후 확장)          # 문집, 승정원일기 등
│   ├── parsed/                 # 파싱된 jsonl (소스별 하위 디렉토리)
│   ├── processed/              # 전처리 중간 결과물 (소스별 하위 디렉토리)
│   └── splits/                 # train / val / test (+ hf_dataset/)
├── scripts/
│   ├── parsers/                # 소스별 파서 (확장 가능)
│   │   └── parse_sillok.py
│   ├── scrape_sillok_korean.py # 국역 수집
│   ├── prepare_pairs.py        # 필터링, 중복 제거, 역자 주석, variant 생성
│   ├── align_and_chunk.py      # 문장 정렬 + sliding window 청킹
│   ├── build_dataset.py        # 분할 + instruction 포맷 + HF Dataset
│   └── evaluate.py             # 평가 (미구현)
├── training/
│   ├── finetune_lora.py        # LoRA 파인튜닝 (미구현)
│   └── configs/                # 학습 하이퍼파라미터
├── inference/
│   └── translate.py            # 추론 (미구현)
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
- 저작권: 학술·연구 목적 이용 가능, 상업적 이용 불가 (국사편찬위원회)

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

### 데이터 특성

- 왕대별 편차 큼: 중종 829만자 ~ 정종 9만자
- 조선 후기(순조, 헌종, 철종) 기록 간소화 경향
- 이본/수정본(선조수정, 숙종보궐정오 등)은 본편보다 기사가 길고 충실
- 짧은 기사(10자 미만) 38,870건(9.4%): 대부분 정형화된 반복 표현 (경연, 천문 등)
  - "○御夕講。" 3,150건, "○御晝講。" 3,069건 등 대량 중복
- 국역의 75%에 괄호 한자 주석 포함: `홍언필(洪彦弼)`
- XML `<index>` 태그에 고유명사 정보 (이름/지명/관직/서명/연호)

## 데이터 파이프라인

### 1. XML 파싱 → jsonl ✅ 완료

`scripts/parsers/parse_sillok.py` → `data/parsed/sillok/articles.jsonl`

```jsonl
{"source": "sillok", "article_id": "waa_10107017_001", "king": "태조", "subset": "태조실록", "date": "태조 1년 7월 28일", "original": "한문 원문...", "title": "기사 제목...", "subjects": [...], "url": "..."}
```

`source` 필드로 데이터 출처를 구분하여 향후 다른 소스 추가 시 통합 가능.

### 2. 국역 수집 🔄 진행중

`scripts/scrape_sillok_korean.py` → `data/parsed/sillok/articles_with_korean.jsonl`

- 일자 단위 배치 요청 (기사 단위 41만 회 → 일자 단위 16만 회, 62% 절감)
- 랜덤 1~5초 대기, 재시도 3회, resume 지원

### 3. 필터링 및 정제 ✅ 완료

`scripts/prepare_pairs.py` → `data/processed/sillok/clean_pairs.jsonl`

- 번역 없는 레코드 제거, `(original, translation)` 해시 기준 중복 제거
- 역자 주석 제거 (strict/relaxed/off): `term(漢字) 짧은설명.` → 설명 부분만 제거
- 괄호 한자 multi-variant 생성 (기사당 1개):
  - clean 50%: `(漢字)` 전체 제거 → instruction "현대 한국어로 번역하라"
  - annotated 30%: `(漢字)` 유지 → instruction "한자를 병기하여 번역하라"
  - mixed 20%: 40%만 랜덤 유지 → instruction "필요한 부분에만 한자를 병기하여 번역하라"

### 4. 문장 정렬 + 청킹 ✅ 완료

`scripts/align_and_chunk.py` → `data/processed/sillok/chunked_pairs.jsonl`

- 토크나이저(`google/translategemma-4b-it`) 기반 토큰 수 계산
- 2048 토큰 이하: 그대로 통과
- 2048 토큰 초과: 한문 `。` / 국역 `. ! ?` 기준 문장 분할 → 정렬 → sliding window 청킹
- overlap: 이전 청크의 마지막 2문장쌍을 context로 포함 (학습 시 loss 미적용)

### 5. 데이터 분할 ✅ 완료

`scripts/build_dataset.py` → `data/splits/{train,val,test}.jsonl` + `hf_dataset/`

실록 데이터는 왕대(王代) 단위로 split:
- **train:** 태조 ~ 성종 (king_code: aa~ia)
- **val:** 연산군 ~ 명종 (ja~ma)
- **test:** 선조 ~ 철종 (na~ya)
- **제외:** 고종/순종 (za~zc)

Gemma 3 턴 구조 + 한국어 instruction으로 포맷. `--save-hf`로 HuggingFace DatasetDict 저장.

Packing은 학습 시점에 DataCollator로 처리 (전처리 범위 밖).

## 학습 계획

### Step 1: Baseline 평가

파인튜닝 전 zero-shot / few-shot 성능 확인 (샘플 50~100건).

### Step 2: LoRA 파인튜닝

- 방법: LoRA (PEFT) 또는 Unsloth
- 모델: 12b 우선
- Instruction: 한국어 ("다음 조선시대 한문을 현대 한국어로 번역하라."), variant별 3종

### Step 3: 평가

- 자동: BLEU, chrF, COMET
- 정성: 관직명, 제도 용어, 인명 등 도메인 용어 정확도

### Step 4: 데모

Gradio 웹 UI. 입력: 한문 원문 → 출력: 현대 한국어 번역.

## 주의사항

- 고전 한문 ≠ 현대 중국어 (문법·어휘 상이, 조선 특유 용어 다수)
- TranslateGemma의 고전 한문 지원 범위 확인 필요
- 국역 번역은 1968~1993년 작업물이라 일부 표현이 현대 기준으로 어색할 수 있음
- 향후 데이터 소스 추가 시 도메인(장르) 간 문체 차이 고려

## 코드 스타일

- Python 3.10+
- 타입 힌트 사용
- docstring 작성
- 경로: pathlib
- 설정: argparse 또는 yaml config
