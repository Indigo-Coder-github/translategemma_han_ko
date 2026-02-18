# 관련 연구 및 리서치 노트

## LoRA vs CPT 비교 (2026-02-09 조사)

### 결론: LoRA SFT 우선, CPT는 결과 보고 판단

### 핵심 논문

#### 1. LoRA Learns Less and Forgets Less (2024, Biderman et al.)
- **링크**: https://hf.co/papers/2405.09673
- LoRA vs full fine-tuning을 SFT/CPT 두 레짐에서 비교
- LoRA는 full FT보다 학습량 적지만, **base 모델 능력을 잘 보존** (정규화 효과)
- full FT의 weight perturbation rank가 LoRA 설정보다 10~100배 높음
- **시사점**: TranslateGemma의 기존 zh→ko 능력 보존에 LoRA가 유리

#### 2. LoRA vs Full Fine-tuning: An Illusion of Equivalence (2024, Shuttleworth et al.)
- **링크**: https://hf.co/papers/2410.21228
- LoRA FT 모델에 "intruder dimensions" 발생 — 높은 rank의 새 singular vector
- intruder dimensions → pre-training 분포에서 멀어짐, 순차 적응 시 불안정
- **대응**: rank-stabilized LoRA 또는 높은 rank 사용으로 완화 가능

#### 3. D-CPT Law (2024, Que et al.)
- **링크**: https://hf.co/papers/2406.01375
- 도메인 특화 CPT의 최적 general:domain 데이터 혼합 비율을 scaling law로 예측
- 6개 도메인(수학, 코드 등)에서 실험, 모델 크기와 데이터 크기에 따른 최적 비율 제시
- **시사점**: CPT에는 대규모 general corpus 리플레이가 필수, 도메인 데이터만으로는 망각 발생
- 실록 원문 ~1억 토큰은 CPT 도메인 코퍼스로도 소규모, general corpus까지 합치면 수백억 필요

#### 3-1. CMR Scaling Law (2024, Gu et al.)
- **링크**: https://hf.co/papers/2407.17467
- CPT에서 general/domain 데이터의 Critical Mixture Ratio(CMR)를 정의
- loss, 혼합 비율, 학습 토큰 수 사이의 power-law 관계 발견
- CMR 이하로 general corpus를 줄이면 catastrophic forgetting 발생
- **시사점**: CPT 시 general corpus 비중이 매우 높아야 하며, 우리 규모에선 비용 과다

#### 4. ADEPT: Continual Pretraining via Adaptive Expansion (2025, Zhang et al.)
- **링크**: https://hf.co/papers/2510.10071
- CPT에서 catastrophic forgetting 문제를 실험적으로 문서화
- full-parameter CPT 대비 ADEPT(선택적 레이어 확장 + 비대칭 학습률)로
  general 도메인 5.76%, target 도메인 5.58% 개선 (15% 파라미터만 튜닝)
- **시사점**: 일반적인 CPT는 기존 능력 저하 위험이 실증됨, 특수 기법 없이는 위험

### 고전 한문 번역 관련

#### 5. WenyanGPT (2025, arXiv 2504.20609)
- LLaMA3-8B-Chinese 기반, CPT + instruction FT로 고전 한문 특화
- **우리와 다른 점**: 범용 LLM을 번역 특화시킨 게 아닌, 이해/생성 전반 목적
- CPT 데이터로 대규모 고전 한문 코퍼스 사용 (우리보다 훨씬 큰 규모)

#### 6. Multi-Agent Classical Chinese Translation (Nature, 2025)
- **링크**: https://www.nature.com/articles/s41598-025-23904-0
- Instruction fine-tuning만으로 BLEURT, BLEU-1, METEOR 18.8~25.7% 향상
- CPT 없이 SFT만으로 충분한 성과 → **우리 접근법 지지**

#### 7. Ancient Chinese Translation LLM (Nature, 2025)
- **링크**: https://www.nature.com/articles/s40494-025-01697-9
- 1.2M 고전-백화문 쌍으로 command fine-tuning → 최고 성능
- CPT 불필요하다는 추가 증거

#### 8. SemiAdapt/SemiLoRA (2025, McGiff & Nikolov)
- **링크**: https://hf.co/papers/2510.18725
- 저자원 언어(아일랜드어) NMT에 LoRA 기반 도메인 적응
- SemiLoRA가 full model fine-tuning과 동등하거나 상회
- **시사점**: 도메인 특화 번역에서 LoRA의 효과 추가 확인

#### 9. ExPLoRA (2024, Khanna et al.)
- **링크**: https://hf.co/papers/2406.10973
- ViT 대상이지만 CPT+LoRA 하이브리드 접근법 제시
- 1~2 블록 unfreeze + 나머지 LoRA → full pre-training 상회
- **참고**: 텍스트 모델에도 유사 전략 가능 (Phase 2 후보)

### 기존 연구
- **H2KE (Son et al., EMNLP 2022)**: 동일 실록 데이터, mBART 기반, 한문→구역→현대한국어 2단계
- **Khayrallah & Koehn (2018)**: 타겟에 소스 언어 복사가 NMT에 가장 치명적
- **Don't Just Scratch the Surface (IJCNLP 2019)**: 한자 주석이 의미 구분에 도움

## 학습 전략 단계별 계획

```
Phase 1: LoRA SFT (2~3일)
  12B, rank 64~128, bf16, L40s 1장
  → BLEU > 25~30 → 성공
  → BLEU 20~25  → Phase 2
  → BLEU < 20   → Phase 3

Phase 2: Enhanced LoRA
  - rank 올리기 (256)
  - RAG (조선시대 용어 사전)
  - Curriculum learning (짧은→긴 텍스트)

Phase 3: Micro-DAPT (CPT light)
  - 1~5B 토큰 혼합(고전+현대) 소규모 CPT
  - 3~5일, 4B 모델로 먼저 검증
  - 효과 확인 후 12B 확대
```

## LoRA 하이퍼파라미터 참고치
- rank: 64~128 (시작), 필요시 256
- lr: 1e-4 ~ 5e-4 (여러 연구에서 5e-4 최적)
- epochs: 2~3
- seq_len: 2048
- batch: GPU 메모리 허용 최대
- precision: bf16 (fp16 금지 - Gemma 3 NaN 문제)
