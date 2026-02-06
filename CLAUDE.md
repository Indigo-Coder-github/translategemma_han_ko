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
│   │   ├── sillok/             # 조선왕조실록
│   │   └── (향후 확장)          # 문집, 승정원일기 등
│   ├── parsed/                 # 파싱된 jsonl (소스별 하위 디렉토리)
│   ├── aligned/                # 문장 정렬된 병렬 쌍
│   └── splits/                 # train / val / test
├── scripts/
│   ├── parsers/                # 소스별 파서 (확장 가능)
│   │   └── parse_sillok.py
│   ├── align_sentences.py      # 문장 정렬
│   ├── filter_and_split.py     # 필터링 및 데이터 분할
│   └── evaluate.py             # 평가
├── training/
│   ├── finetune_lora.py        # LoRA 파인튜닝
│   └── configs/                # 학습 하이퍼파라미터
├── inference/
│   └── translate.py            # 추론
├── demo/
│   └── app.py                  # Gradio 데모 UI
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

형식: XML (원문 + 국역 포함), 인코딩: UTF-8
스키마: 한국사데이터베이스 DTD v1.3

XML 파일 규칙:
- 왕별 파일 분리, 파일명 가운데 영문자 = 왕조 순서 (a=태조, d=세종)
- 끝 숫자 = 재위년도
- `_000.xml` = 총서, `_200~` = 별첨/부록
- 고유명사에 REF 속성으로 ID 부여

### Phase 2 이후: 확장 후보 (미착수)

- **한국고전종합DB (db.itkc.or.kr):** 문집, 상소문 등 장르 다양. 단, Open API는 메타데이터 수준이고 크롤링은 저작권 문제 있음. 별도 데이터 제공 요청 또는 공모전 활용 등 검토 필요.
- **승정원일기:** 국역 미완성. 기계번역 결과만 있어 학습 데이터로 부적합.

## 데이터 파이프라인

### 1. XML 파싱 → jsonl

소스별 파서를 `scripts/parsers/`에 분리. 기사(article) 단위 추출.

```jsonl
{"source": "sillok", "subset": "태조실록", "volume": "1권", "date": "태조 1년 7월 28일", "original": "한문 원문...", "translation": "현대 한국어...", "url": "..."}
```

`source` 필드로 데이터 출처를 구분하여 향후 다른 소스 추가 시 통합 가능.

### 2. 문장 정렬 (Sentence Alignment)

- 한문: 마침표(。) 기준 분리
- 국역: 문장 부호(. !) 기준 분리
- 순서 기반 1:1 매칭 → 불일치 시 Gale-Church 등 동적 정렬
- 신뢰도 낮은 쌍은 제외 또는 수동 검토 대상

### 3. 필터링

- 원문 5자 미만 / 번역 10자 미만 → 제거
- 과도하게 긴 쌍 → 분할
- 중복 제거
- 역자 주, 편집 주석 등 노이즈 처리

### 4. 데이터 분할

실록 데이터는 왕대(王代) 단위로 split:
- **train:** 태조 ~ 성종
- **val:** 연산군 ~ 명종
- **test:** 선조 ~ 철종

비율은 실제 분량 확인 후 조정. 향후 다른 소스 추가 시 소스 단위 분할도 고려.

## 학습 계획

### Step 1: Baseline 평가

파인튜닝 전 zero-shot / few-shot 성능 확인 (샘플 50~100건).

### Step 2: LoRA 파인튜닝

- 방법: LoRA (PEFT) 또는 Unsloth
- 모델: 12b 우선
- 고전 한문임을 명시하는 instruction 설계 필요

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
