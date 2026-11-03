# ParkIN Soon (박인순)

**멀티모달 파킨슨병 조기 선별 키오스크**

후각(B-SIT 12항목) · 자율신경 증상(SCOPA-AUT 변비 3항목) · 나선 손그림(이미지 + 운동학) 세 가지 신호를 각각 학습한 모델로 평가하고, 결과를 노인 사용자가 이해할 수 있는 언어로 설명해주는 Streamlit 애플리케이션입니다. 결과 화면에는 RAG 기반 AI 상담 캐릭터 "박인순"이 함께 제공됩니다.

> ⚕️ **본 프로젝트는 연구 및 프로토타입 목적입니다. 의학적 진단을 대체하지 않으며, 진단 도구로 사용해서는 안 됩니다.** 자세한 내용은 [알려진 한계](#알려진-한계)를 반드시 확인해주세요.

---

## 검사 흐름

```
Step 0  고객 정보 입력 (이름 / 전화번호)  →  과거 검사 기록 조회
Step 1  후각 인지 검사       12개 향 스크래치 카드, 구별됨/모르겠음
Step 2  배변 상태 체크       SCOPA-AUT 5·6·7번 문항 (0~3점)
Step 3  나선 그리기          400×400 캔버스, 점선 가이드 따라 그리기
Step 4  결과
        ├─ Page 1  후각 · 배변 결과 + 정상군 비교 그래프
        ├─ Page 2  손그림 시각화(AI 주목 영역) + 운동학 수치 + 신경과 지도
        └─ Page 3  최종 레포트 + 이전 검사 비교 + AI 박인순 상담(RAG)
```

각 단계마다 캐릭터가 안내 문구를 제공하고, 브라우저 `speechSynthesis` API로 음성 안내(ko-KR)를 재생할 수 있습니다.

---

## 모델 구성

| 모달 | 입력 | 모델 | 학습 데이터 | 성능 | 임계값 |
|---|---|---|---|---|---|
| **후각+배변** (`p_olf`) | B-SIT 12개 이진값 + SCOPA 3문항 (15차원) | BorderlineSMOTE → XGBoost (200 trees) | PPMI 944명 (PD 809 / HC 135) | 5-Fold CV AUC **0.775** | 0.4617 |
| **그림 모양** (`p_img`) | 224×224 grayscale→3ch 나선 이미지 | MobileNetV2 (ImageNet 전이학습) | Spiral 72장 학습 / 30장 holdout | Holdout AUC **0.8133** | 0.4510 |
| **손 움직임** (`p_kin`) | velocity/acceleration/jerk 통계 12개 | StandardScaler → RandomForest (300 trees, balanced) | UCI 40명 + PaHaW 74명 = 114명 | 5-Fold CV AUC **0.7615** | 0.5418 |

### 종합 판정 (AUC 가중 Late Fusion)

세 모달은 서로 다른 데이터셋(PPMI / Kaggle Spiral / UCI+PaHaW)에서 학습되어 환자 단위로 특성을 합칠 수 없으므로, 모델별로 따로 예측한 뒤 결과를 합치는 Late Fusion을 사용합니다.

- **융합 점수** = Σ wᵢ · pᵢ (가중치 wᵢ는 각 모델 AUC를 정규화한 값: olf .330 / img .346 / kin .324)
- **융합 임계값** = Σ wᵢ · Tᵢ ≈ **0.484**
  - 모델마다 확률 스케일과 임계값이 달라 확률만 평균내면 왜곡되므로, 같은 가중치로 임계값도 합성해 비교합니다.

| 조건 | 결과 | 안내 |
|---|---|---|
| 임계값을 넘은 모델 0개 | ✅ 특이 징후 없음 | 정기 검진 권장 |
| 1개 이상이지만 융합 점수 < 융합 임계값 | ⚠️ 주의 필요 | 전문의 상담 권장 |
| 융합 점수 ≥ 융합 임계값 | 🔴 이상 징후 | 전문의 소견 권유 + 근처 신경과 지도 안내 |

이상 항목이 0개면 융합 점수는 항상 융합 임계값 미만이고, 융합 점수가 임계값 이상이면 항상 1개 이상이 이상이므로 화면의 항목별 표시와 종합 판정이 서로 모순되지 않습니다. 상담 챗봇에도 이 융합 점수가 전달됩니다.

---

## 프로젝트 구조

```
ParkIN-Soon/
├── kiosk.py                    # 메인 Streamlit 앱 (Step 0~4 전체)
├── ParkinsoonAI.py             # 캐릭터 안내 UI + 결과 페이지 챗봇 렌더러
├── rag_chatbot.py              # 로컬 벡터DB 검색 + GPT 답변 생성 (CLI 겸용)
│
├── preprocessing.ipynb         # PPMI 원본 → B-SIT 12 + SCOPA 병합/정제
├── PPMI.ipynb                  # 후각+배변 모델 비교 및 XGBoost 최종 학습
├── drawing_pipeline.ipynb      # MobileNetV2 나선 이미지 분류 (5-Fold + Holdout)
├── kinematic_pipeline.ipynb    # UCI+PaHaW 운동학 RandomForest
├── late_fusion.ipynb           # AUC 기반 융합 가중치 산출
│
├── modeling/
│   ├── olf_con_model.pkl           # 후각+배변 XGBoost 파이프라인
│   ├── drawing_cnn_model.h5        # 나선 이미지 CNN
│   ├── drawing_kinematic_model.pkl # 운동학 RF ({'pipeline': ...})
│   ├── fusion_config.json          # AUC 기반 융합 가중치
│   ├── vector_db.pkl               # RAG 논문 임베딩 DB
│   └── customer_records.db         # 검사 기록 SQLite  ⚠️ 커밋 금지 대상
│
├── data/PPMI_BSIT12_SCOPA.csv  # 전처리 완료 학습 데이터
├── parkinsons_drawings/        # spiral/wave × training/testing × healthy/parkinson
├── hospital_data.csv / .xlsx   # 전국 요양기관 (신경과 필터링용)
│
└── (필요하지만 저장소에 없음)
    ├── character.png           # 캐릭터 이미지 — 없으면 플레이스홀더로 대체됨
    ├── .env                    # OPENAI_API_KEY
    └── LLM논문/                # RAG DB 재생성용 PDF 원본
```

---

## 실행 방법

### 1. 환경

Python 3.10 권장. **Windows 기준으로 개발되었습니다** (한글 폰트 `Malgun Gothic` 하드코딩).

```bash
pip install streamlit streamlit-drawable-canvas streamlit-folium folium
pip install tensorflow keras scikit-learn xgboost imbalanced-learn joblib
pip install opencv-python pillow matplotlib seaborn pandas numpy openpyxl
pip install openai sentence-transformers pypdf
```

> `requirements.txt`가 아직 없습니다. `kiosk.py`에는 Streamlit/Keras 버전 차이를 흡수하기 위한 몽키패치(`st_image.image_to_url`, `CompatInputLayer`)가 들어 있어 **버전 조합에 민감합니다.** 동작하는 조합을 찾으면 `pip freeze > requirements.txt`로 고정해두시길 권합니다.

### 2. API 키 (AI 상담 기능용)

프로젝트 루트에 `.env` 생성:

```
OPENAI_API_KEY=sk-...
```

키가 없어도 검사·결과 화면은 정상 동작하며, 상담 챗봇만 비활성화됩니다.

### 3. 실행

```bash
streamlit run kiosk.py
```

`MODEL_DIR` 환경변수로 모델 폴더 위치를 바꿀 수 있습니다 (기본값: `./modeling`).

### 4. RAG 벡터 DB 재생성 (선택)

`modeling/vector_db.pkl`이 이미 포함되어 있습니다. 논문을 교체·추가하려면 루트에 `LLM논문/` 폴더를 만들고 PDF를 넣은 뒤:

```bash
python rag_chatbot.py --build
```

검색 동작만 확인하려면:

```bash
python rag_chatbot.py --query "파킨슨 후각 저하"
```

임베딩은 로컬 모델 `jhgan/ko-sroberta-multitask`를 사용하며, 답변 생성만 OpenAI `gpt-4o-mini`를 호출합니다.

---

## 데이터 출처

- **PPMI** (Parkinson's Progression Markers Initiative) — UPSIT 후각 검사, SCOPA-AUT 자율신경 설문, Participant Status. 원본 CSV는 라이선스상 저장소에 포함되지 않으며, `preprocessing.ipynb`가 `./PPMI/` 경로를 기대합니다.
  - B-SIT 12항목 중 원본의 Paint Thinner / Turpentine은 PPMI에 없어 **Cherry / Dill Pickle로 대체**했습니다.
  - Prodromal(COHORT 4), SWEDD(3)은 제외하고 PD(1) / HC(2)만 사용했습니다.
- **UCI HandPD** — Spiral 태스크(TestID 0) 좌표·타임스탬프.
- **PaHaW** — Task 8(나선) `.svc` 파일 중 필기 접촉 구간(`ButtonStatus == 1`)만 사용.
- **Kaggle Parkinson's Drawings** — `parkinsons_drawings/` (spiral, wave). 현재 **spiral만 사용**합니다.
- **건강보험심사평가원 요양기관 정보** — `hospital_data.csv`. 기관명에 "신경과"가 포함된 행만 필터링합니다.

---

## AI 상담 (박인순) 설계

`rag_chatbot.py`의 시스템 프롬프트는 의료 맥락에 맞춰 다음을 강제합니다.

- **금지 표현**: "파킨슨병입니다", "파킨슨병이 확실합니다", "앞으로 파킨슨병에 걸립니다"
- 한국어 해요체, 짧은 문장, 전문용어 최소화 (노인 사용자 대상)
- 답변 구조: `[1] 결과 요약 → [2] 쉬운 해석 → [3] 다음 행동 → [4] 보호자/보건소용 요약`
- 논문 근거는 내부 참고용으로만 사용하고 출처는 답변에 노출하지 않음

또한 **불안 표현 선제 차단**이 있습니다. 질문에 "죽", "큰일", "무서", "위험한가" 등이 포함되면 LLM 호출 없이 고정 안심 문구를 반환하고, 응급 증상(편측 마비, 언어 장애, 의식 저하 등) 시 119 연락을 안내합니다. API 호출 실패 시에도 동일한 성격의 폴백 문구를 반환합니다.

---

## 알려진 한계

**이 섹션의 항목들은 결과 수치의 해석에 직접 영향을 줍니다.**

### 모델·수치 관련

1. **운동학 타임스탬프가 실측이 아닙니다.** 캔버스 경로 좌표에 16ms 간격을 일괄 부여하므로(`kiosk.py:1200`), 속도·가속도·jerk가 사실상 "점 간 거리"의 파생값입니다. 모델은 태블릿 실측 타임스탬프로 학습되었으므로 도메인이 일치하지 않습니다. **`p_kin`과 화면의 `px/s` 수치는 참고 이상으로 해석하면 안 됩니다.**
2. **운동학 모델의 소스별 정규화가 추론에 적용되지 않습니다.** 학습 시 UCI/PaHaW별 `StandardScaler`를 적용했고 `src_scalers`도 저장했으나, 키오스크는 원시 특성을 파이프라인에 직접 넣습니다.
3. ~~Late Fusion 가중치가 판정에 반영되지 않습니다.~~ → **해결**: 융합 점수·융합 임계값으로 판정합니다.
4. ~~챗봇의 "종합 위험도"가 `cnt/3`입니다.~~ → **해결**: 융합 점수를 전달합니다. 단, 모델들이 확률 보정(calibration)되지 않았으므로 **진단 확률이 아닌 상대 점수**입니다.
5. ~~손그림 특성 계산 실패 시 `[0.5]×12`를 모델에 투입~~ → **해결**: 계산 실패 시 "측정 불가"로 표시하고, 이상 항목 수와 융합 점수에서 제외합니다(남은 모달 가중치를 재정규화).
6. ~~"Grad-CAM"으로 표기~~ → **수정**: 실제 방식대로 입력 saliency map으로 표기합니다. Grad-CAM으로의 교체는 개선 과제입니다.
7. ~~후각 검사 기본값이 "구별됨"~~ → **해결**: 후각·배변 문항 모두 기본 선택값을 없애고, 미응답 문항이 있으면 다음 단계로 넘어가지 않습니다.
8. **임계값(0.4617 / 0.4510 / 0.5418)의 산출 코드가 저장소에 없습니다.** UI에는 Youden's Index 기반으로 안내되어 있습니다.

### 데이터 관련

9. **PPMI 클래스 불균형**: PD 809명 vs HC 135명 (약 6:1). 실제 선별 환경의 유병률과 정반대이므로 `p_olf`는 구조적으로 높게 나옵니다.
10. **손그림 데이터 규모가 작습니다**: spiral 학습 72장, holdout 30장. Holdout AUC 0.8133의 신뢰구간이 매우 넓습니다. `wave` 데이터는 미사용입니다.
11. **이미지 도메인 시프트**: 학습 데이터는 종이에 펜으로 그린 나선 사진, 입력은 400px 캔버스의 3px 디지털 선입니다. 224px 리사이즈 후 선 굵기와 질감이 크게 다릅니다.

### 운영 관련

12. **개인정보가 평문으로 저장됩니다.** 이름·전화번호가 `modeling/customer_records.db`에 암호화 없이 저장되며, **이 파일이 현재 git에 커밋되어 있습니다.** `.gitignore`가 없습니다. 실제 검사자 데이터를 다루기 전에 반드시 제외 처리해야 합니다.
13. **과거 기록 조회 인증이 이름+전화번호뿐입니다.** 두 값만 알면 타인의 결과를 조회할 수 있습니다.
14. **사용자 입력이 이스케이프 없이 HTML에 삽입됩니다** (`unsafe_allow_html=True`). 자기 세션 한정이지만 처리가 필요합니다.
15. **Windows 종속**: 한글 폰트가 `Malgun Gothic`으로 고정되어 있어 Linux 배포 시 그래프 한글이 깨집니다.
16. **결과 페이지 rerun마다 CNN 예측과 시각화를 재계산합니다.** 챗봇 질문 한 번마다 전체가 다시 돌아 응답이 느립니다.

---

## 개선 로드맵

우선순위 순:

- [ ] 캔버스 포인터 이벤트에서 **실제 타임스탬프 수집** (한계 1) — 이것이 해결되지 않으면 운동학 모달 전체의 의미가 제한적입니다
- [ ] `src_scalers` 적용 또는 웹 캔버스 데이터로 운동학 모델 재학습 (한계 2)
- [x] 특성 계산 실패 시 `None` 반환 → "측정 불가" 표기, `cnt` 계산에서 제외 (한계 5)
- [ ] `.gitignore` 추가 + `customer_records.db` 히스토리에서 제거, 전화번호 해시 저장 (한계 12)
- [ ] `requirements.txt` 버전 고정
- [x] `fusion_config.json` 가중치를 실제 판정에 반영하고, 챗봇 `final_risk`를 융합 점수로 교체 (한계 3, 4)
- [ ] 모델 확률 보정(Platt / Isotonic) 후 융합해 융합 점수를 확률로 해석 가능하게 만들기
- [ ] `out_relu` 기반 실제 Grad-CAM으로 교체 (한계 6) — `drawing_pipeline.ipynb`에 구현 있음
- [ ] 후각 라디오 기본값을 미선택으로 변경 (한계 7)
- [ ] 임계값 산출 코드를 `late_fusion.ipynb`에 포함
- [ ] `hospital_data` 중복 파일 정리 (csv 25MB + xlsx 17MB → 신경과 행만 추출)
- [ ] 결과 페이지 예측 결과 캐싱, `ParkinsonRAG` 인스턴스 `st.cache_resource` 적용

---

## 라이선스

미정. PPMI · UCI HandPD · PaHaW 데이터는 각 기관의 이용 약관을 따르며, 원본 데이터는 저장소에 포함되어 있지 않습니다.
