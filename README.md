# 한문 번역기 (Hanja Translator)

조선시대 한문(漢文) → 현대 한국어 번역 모델.
TranslateGemma를 한문-한국어 병렬 코퍼스에 파인튜닝합니다.

## 개요

조선왕조실록을 시작으로 다양한 조선시대 한문 사료의 병렬 데이터를 활용하여, Google TranslateGemma 모델을 고전 한문 번역에 특화시키는 프로젝트입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 데이터

현재 [공공데이터포털](https://www.data.go.kr/data/15053647/fileData.do)의 조선왕조실록 XML 데이터를 사용합니다. (이용허락 제한 없음)

> `data/` 디렉토리는 git 추적에서 제외됩니다. 아래 절차에 따라 직접 준비하세요.

### 데이터 준비

```bash
# 1. 공공데이터포털에서 XML 다운로드 후 data/raw/sillok/에 배치
mkdir -p data/raw/sillok data/parsed data/processed data/splits

# 2. XML 파싱
python scripts/parsers/parse_sillok.py

# 3. 국역 수집 (sillok.history.go.kr API, 수일 소요)
python scripts/scrape_sillok_korean.py --delay-min 1 --delay-max 5

# 4. 필터링 + 역자 주석 제거 + variant 생성
python scripts/prepare_pairs.py

# 5. 문장 정렬 + 청킹
python scripts/align_and_chunk.py

# 6. 분할 + instruction 포맷 + HF Dataset
python scripts/build_dataset.py --save-hf
```

데이터 스키마 상세는 [`data/README.md`](data/README.md) 참조.

## 모델

| 모델 | 용도 | HuggingFace |
|---|---|---|
| TranslateGemma 4B | 로컬 테스트 | `google/translategemma-4b-it` |
| TranslateGemma 12B | 파인튜닝 1순위 | `google/translategemma-12b-it` |
| TranslateGemma 27B | 대형 실험 | `google/translategemma-27b-it` |

## 학습

```bash
# LoRA 파인튜닝
python training/finetune_lora.py --config training/configs/default.yaml

# Multi-GPU (FSDP2)
accelerate launch training/finetune_lora.py --config training/configs/default.yaml
```

## 데모

```bash
python demo/app.py
```

## 환경

- Python 3.10+
- GPU: L40s x 2 (학습), RTX 3060Ti (로컬 테스트)
- 주요 의존성: transformers, peft, accelerate, datasets, gradio

## 라이선스

- 코드: MIT
- 모델: [Gemma License](https://ai.google.dev/gemma/terms)
- 데이터: 공공데이터포털 이용허락 제한 없음
