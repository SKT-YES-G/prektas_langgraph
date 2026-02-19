# Pre-KTAS LangGraph

STT 또는 키보드 입력을 받아 **LangGraph 기반 Multi-stage KTAS 분류**를 수행하는 FastAPI 서버입니다.

---

## 개요

Pre-KTAS는 응급실 내원 환자의 중증도를 자동으로 분류하는 AI 시스템입니다.
환자의 진술·활력징후를 입력받아 KTAS Stage 2 → 3 → 4 순서로 계층적으로 분류하고,
각 단계의 **분류 근거(evidence_spans)** 를 함께 반환합니다.

---

## 시스템 구조

```
[텍스트 입력 (STT / 키보드)]
        │
        ▼
  retriage_judge          ← 새 정보를 보고 어느 단계부터 재평가할지 결정
        │
  retriage_target
        │
   ┌────┼────┐
   ▼    ▼    ▼
 s2   s3   s4            ← 해당 stage 재평가 노드
   │    │    │
   ├─추가질문─┤ → ask_question → [END]
   │    │    │
   └────┴────┘
        │ 재분류
        ▼
  stage2_classifier
        │
  stage3_classifier
        │
  stage4_classifier → [END]
```

### KTAS 분류 계층

| 단계 | 설명 | 예시 |
|------|------|------|
| Stage 2 | 주증상 계통 (18개) | 심혈관계, 호흡기계, 신경계 … |
| Stage 3 | 세부 증상 | 가슴통증(심장성), 호흡곤란 … |
| Stage 4 | 구체 임상 상태 | NSTEMI/불안정 협심증, STEMI … |

---

## 설치 및 실행

### 1. 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 입력합니다.

```env
OPENAI_API_KEY=sk-...
```

### 3. 서버 실행

```bash
uvicorn pre_ktas.app:app --reload --port 8000
```

### 4. Swagger UI 접속

```
http://localhost:8000/docs
```

---

## API 엔드포인트

### `POST /triage/input` — 텍스트 입력 → 분류 실행

환자 진술 텍스트를 전송하면 Stage 2 → 3 → 4 분류를 자동으로 수행합니다.
같은 `session_id`로 반복 호출하면 이전 상태에 이어서 **재평가**됩니다.

**Request Body**

```json
{
  "text": "58세 남성. 어제저녁부터 갑자기 가슴이 쥐어짜는 듯이 아프고 왼쪽 팔로 방사됩니다. 혈압 150/95, 맥박 102, SpO2 96%.",
  "source": "stt",
  "session_id": "patient_001"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `text` | string | STT 또는 직접 입력 텍스트 |
| `source` | `"keyboard"` \| `"stt"` | 입력 소스 |
| `session_id` | string | 환자/세션 식별자 (기본값: `"default"`) |

**Response**

```json
{
  "session_id": "patient_001",
  "message": "stage4 분류 완료 → NSTEMI/불안정 협심증 의심",
  "state": {
    "stage2_selection": "심혈관계",
    "stage3_selection": "가슴통증(심장성)",
    "stage4_selection": "NSTEMI/불안정 협심증 의심",
    "classification_log": [
      {
        "stage": 2,
        "selection": "심혈관계",
        "confidence": "높음",
        "evidence_spans": [
          {"quote": "가슴이 쥐어짜는", "interpretation": "심장성 흉통 전형 양상"},
          {"quote": "왼쪽 팔로 방사", "interpretation": "심근허혈 방사통 패턴"}
        ],
        "reason": "전형적인 심장성 흉통 패턴 및 방사통으로 심혈관계로 분류."
      }
    ]
  }
}
```

---

### `GET /triage/state` — 현재 분류 상태 조회

```
GET /triage/state?session_id=patient_001
```

---

### `POST /triage/reset` — 세션 초기화

새 환자 내원 시 호출합니다.

```
POST /triage/reset?session_id=patient_001
```

---

### `GET /health` — 헬스체크

```json
{"status": "ok", "graph_ready": true}
```

---

## 프로젝트 구조

```
pre_ktas/
├── app.py                          # FastAPI 애플리케이션 (엔드포인트)
├── graph/
│   ├── state.py                    # GraphState TypedDict 정의
│   ├── build_graph.py              # 그래프 빌드 및 컴파일
│   └── route.py                    # conditional edge 라우팅 함수
└── nodes/
    ├── data/
    │   └── ktas_candidates.py      # KTAS Stage2→3→4 하드코딩 매핑
    ├── retriage/
    │   ├── judge.py                # 재평가 판단 노드 (어느 stage부터?)
    │   ├── stage2.py               # Stage 2 재평가 (추가질문 | 재분류)
    │   ├── stage3.py               # Stage 3 재평가
    │   └── stage4.py               # Stage 4 재평가
    ├── classify/
    │   ├── stage2.py               # Stage 2 classifier (LLM)
    │   ├── stage3.py               # Stage 3 classifier (LLM)
    │   └── stage4.py               # Stage 4 classifier (LLM)
    └── ask/
        └── question.py             # 추가 질문 포맷 노드
```

---

## 주요 설계

- **세션 지속성**: `MemorySaver` + `session_id`를 LangGraph `thread_id`로 사용하여 동일 환자의 여러 번 입력 사이에서 상태를 유지합니다.
- **Structured Output**: 모든 LLM 호출은 Pydantic 스키마로 강제하여 파싱 오류를 방지합니다.
- **evidence_spans**: 각 분류 단계마다 환자 진술에서 발췌한 근거 표현(`quote`)과 그 임상적 해석(`interpretation`)을 함께 반환합니다.
- **Classifier cascade**: Stage 2 분류 후 항상 Stage 3 → Stage 4 순서로 연쇄 실행됩니다.
