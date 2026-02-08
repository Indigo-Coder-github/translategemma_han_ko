# 데이터 스키마 문서

> 실제 데이터 파일은 `.gitignore`에 의해 추적되지 않습니다.
> 각 단계의 샘플 데이터는 `data/samples/`에서 확인할 수 있습니다.

## 파이프라인 개요

```
data/parsed/sillok/articles_with_korean.jsonl   (국역 수집 결과)
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

---

## Stage 0: XML 파싱 결과

**파일:** `data/parsed/sillok/articles.jsonl`
**생성:** `python scripts/parsers/parse_sillok.py`
**건수:** 414,024건

| 필드 | 타입 | 설명 |
|------|------|------|
| `source` | string | 데이터 출처 (항상 `"sillok"`) |
| `article_id` | string | 기사 고유 ID (예: `waa_10107017_001`) |
| `king_code` | string | 왕 코드 (예: `aa`=태조, `da`=세종) |
| `king` | string | 왕 이름 (예: `"태조"`) |
| `subset` | string | 실록 부분집합 (예: `"太祖實錄 元年"`) |
| `volume` | string | 권/책 정보 |
| `page` | string | 쪽/면 정보 |
| `doc_no` | string | 기사 번호 |
| `date` | string | 원문 날짜 (간지) |
| `date_western` | string | 서력 환산 날짜 (예: `"1392-07-17L0"`) |
| `date_ganji` | string | 간지 날짜 |
| `title` | string | 기사 제목 (현대 한국어) |
| `original` | string | 한문 원문 |
| `subjects` | list[string] | 주제 분류 태그 |
| `url` | string | 실록 웹 URL |

---

## Stage 0.5: 국역 수집 결과

**파일:** `data/parsed/sillok/articles_with_korean.jsonl`
**생성:** `python scripts/scrape_sillok_korean.py`
**샘플:** `data/samples/articles_with_korean.sample.jsonl`

Stage 0의 모든 필드 + 아래 2개 추가:

| 필드 | 타입 | 설명 |
|------|------|------|
| `translation` | string \| null | 현대 한국어 번역 (국역). null이면 국역 미수집 |
| `korean_id` | string \| null | 국역 기사 ID (예: `kaa_10107018_001`) |

---

## Stage 1: 정제된 병렬 쌍

**파일:** `data/processed/sillok/clean_pairs.jsonl`
**생성:** `python scripts/prepare_pairs.py`
**샘플:** `data/samples/clean_pairs.sample.jsonl`

| 필드 | 타입 | 설명 |
|------|------|------|
| `source` | string | 데이터 출처 (`"sillok"`) |
| `article_id` | string | 기사 고유 ID |
| `king_code` | string | 왕 코드 |
| `king` | string | 왕 이름 |
| `original` | string | 한문 원문 |
| `translation` | string | 정제된 한국어 번역 (역자 주석 제거, variant 적용 후) |
| `variant` | string | 괄호 한자 처리 방식: `"clean"` \| `"annotated"` \| `"mixed"` |
| `instruction_type` | string | instruction 유형: `"plain"` \| `"annotated"` \| `"mixed"` |

### variant 설명

| variant | 비율 | 처리 | instruction |
|---------|------|------|-------------|
| `clean` | 50% | `홍언필(洪彦弼)` → `홍언필` | "현대 한국어로 번역하라" |
| `annotated` | 30% | `홍언필(洪彦弼)` 유지 | "한자를 병기하여 번역하라" |
| `mixed` | 20% | 40%만 랜덤 유지 | "필요한 부분에만 한자를 병기하여 번역하라" |

---

## Stage 2: 청킹된 병렬 쌍

**파일:** `data/processed/sillok/chunked_pairs.jsonl`
**생성:** `python scripts/align_and_chunk.py`
**샘플:** `data/samples/chunked_pairs.sample.jsonl`

| 필드 | 타입 | 설명 |
|------|------|------|
| `source` | string | 데이터 출처 |
| `article_id` | string | 원본 기사 ID |
| `chunk_id` | string | 청크 ID (예: `waa_10107017_001_c002`) |
| `king_code` | string | 왕 코드 |
| `king` | string | 왕 이름 |
| `original` | string | 청크 한문 원문 |
| `translation` | string | 청크 한국어 번역 |
| `variant` | string | 괄호 한자 처리 방식 |
| `instruction_type` | string | instruction 유형 |
| `context_original` | string \| null | 이전 청크의 overlap 원문 (첫 청크이면 null) |
| `context_translation` | string \| null | 이전 청크의 overlap 번역 (첫 청크이면 null) |
| `is_first_chunk` | boolean | 기사의 첫 번째 청크 여부 |
| `total_chunks` | integer | 해당 기사의 총 청크 수 (1이면 분할 없음) |
| `token_count` | integer | 토큰 수 (instruction + input + output 포함) |

### 청킹 규칙

- **2048 토큰 이하** (~96%): 분할 없이 통과 (`total_chunks=1`)
- **2048 토큰 초과** (~4%): 문장 단위 분할 + sliding window
  - 원문: `。` (구점) 기준 분할
  - 국역: `. ! ?` 기준 분할
  - 기본 청크 크기: 3~5 문장쌍
  - overlap: 이전 청크의 마지막 2 문장쌍을 context로 포함

---

## Stage 3: 최종 학습 데이터

**파일:** `data/splits/{train,val,test}.jsonl`
**생성:** `python scripts/build_dataset.py`
**샘플:** `data/samples/train.sample.jsonl`

| 필드 | 타입 | 설명 |
|------|------|------|
| `article_id` | string | 원본 기사 ID |
| `chunk_id` | string | 청크 ID |
| `king_code` | string | 왕 코드 |
| `split` | string | 데이터 분할: `"train"` \| `"val"` \| `"test"` |
| `instruction` | string | 한국어 instruction 텍스트 |
| `input` | string | 한문 원문 (번역 대상) |
| `output` | string | 한국어 번역 (정답) |
| `context_input` | string \| null | 맥락 원문 (이전 청크 overlap) |
| `context_output` | string \| null | 맥락 번역 |
| `formatted` | string | Gemma 3 턴 구조로 완성된 전체 프롬프트 |
| `token_count` | integer | 토큰 수 |
| `variant` | string | 괄호 한자 처리 방식 |

### 데이터 분할 기준 (왕대)

| split | 왕대 | king_code |
|-------|------|-----------|
| train | 태조 ~ 성종 | `aa` ~ `ia` |
| val | 연산군 ~ 명종 | `ja` ~ `ma` |
| test | 선조 ~ 철종 | `na` ~ `ya` |
| (제외) | 고종, 순종 | `za` ~ `zc` |

### Gemma 3 instruction 포맷

**기본 (context 없음):**
```
<bos><start_of_turn>user
다음 조선시대 한문을 현대 한국어로 번역하라.

{input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>
```

**context 있음 (2번째 이후 청크):**
```
<bos><start_of_turn>user
다음 조선시대 한문을 현대 한국어로 번역하라.

[맥락 - 이전 문장]
원문: {context_input}
번역: {context_output}

[번역할 문장]

{input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>
```

### instruction 변형

| variant | instruction |
|---------|-------------|
| `plain` (clean) | "다음 조선시대 한문을 현대 한국어로 번역하라." |
| `annotated` | "다음 조선시대 한문을 한자를 병기하여 현대 한국어로 번역하라." |
| `mixed` | "다음 조선시대 한문을 필요한 부분에만 한자를 병기하여 현대 한국어로 번역하라." |

---

## HuggingFace Dataset

**디렉토리:** `data/splits/hf_dataset/`
**생성:** `python scripts/build_dataset.py --save-hf`

`DatasetDict` (arrow 형식)으로 저장. `train`, `val`, `test` split 포함.
`data/splits/{train,val,test}.jsonl`과 동일한 스키마.
