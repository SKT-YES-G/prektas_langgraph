"""
Pre-KTAS FastAPI 애플리케이션

실행:
    uvicorn pre_ktas.app:app --reload --port 8000

Swagger UI:
    http://localhost:8000/docs

엔드포인트:
    POST /triage/input   - STT·키보드 텍스트 수신 → 그래프 실행 → 분류 결과 반환
    GET  /triage/state   - 현재 분류 상태 조회
    POST /triage/reset   - 세션 초기화
    GET  /health         - 헬스체크
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from pre_ktas.graph.build_graph import build_graph, get_initial_state


# ── 더미 데이터 예시 (Swagger 예시 / 테스트용) ────────────────────────────────
_DUMMY_INPUT_EXAMPLE = {
    "text": (
        "환자 58세 남성. 어제저녁부터 갑자기 가슴이 쥐어짜는 듯이 아프고 "
        "왼쪽 팔로 방사됩니다. 식은땀이 흐르고 숨이 차요. "
        "혈압 150/95, 맥박 102, SpO2 96%."
    ),
    "source": "stt",
    "session_id": "patient_001",
}

_DUMMY_STATE_EXAMPLE = {
    "session_id": "patient_001",
    "stage2_selection": "심혈관계",
    "stage3_selection": "가슴통증(심장성)",
    "stage4_selection": "NSTEMI/불안정 협심증 의심",
    "retriage_target": "stage2",
    "retriage_action": "재분류",
    "additional_questions": [],
    "classification_log": [
        {
            "stage": 2,
            "selection": "심혈관계",
            "confidence": "높음",
            "evidence_spans": [
                {"quote": "가슴이 쥐어짜는", "interpretation": "심장성 흉통 전형 양상"},
                {"quote": "왼쪽 팔로 방사", "interpretation": "심근허혈 방사통 패턴"},
            ],
            "reason": "전형적인 심장성 흉통 패턴 및 방사통으로 심혈관계로 분류.",
        },
        {
            "stage": 3,
            "selection": "가슴통증(심장성)",
            "confidence": "높음",
            "evidence_spans": [
                {"quote": "가슴이 쥐어짜는", "interpretation": "심장성 흉통 직접 기술"},
                {"quote": "식은땀이 흐르고", "interpretation": "자율신경계 반응, 급성관상동맥증후군 시사"},
            ],
            "reason": "흉통 성상과 동반 증상이 심장성임을 지지.",
        },
        {
            "stage": 4,
            "selection": "NSTEMI/불안정 협심증 의심",
            "confidence": "중간",
            "evidence_spans": [
                {"quote": "SpO2 96%", "interpretation": "산소포화도 경계 정상, STEMI보다 경증"},
                {"quote": "맥박 102", "interpretation": "빈맥으로 심근 허혈성 반응 시사"},
            ],
            "reason": "ST 상승 기술 없고 활력징후 부분 안정으로 NSTEMI 가능성.",
        },
    ],
}


# ── Pydantic 요청/응답 모델 ───────────────────────────────────────────────────

class InputRequest(BaseModel):
    text: str = Field(
        ...,
        description="STT 또는 키보드로 입력된 환자 관련 텍스트",
        examples=[_DUMMY_INPUT_EXAMPLE["text"]],
    )
    source: Literal["keyboard", "stt"] = Field(
        default="keyboard",
        description="입력 소스: 'keyboard' (직접 입력) 또는 'stt' (음성 인식)",
        examples=["stt"],
    )
    session_id: str = Field(
        default="default",
        description="환자/세션 식별자. 같은 session_id로 호출하면 이전 상태에 이어 분류됨.",
        examples=["patient_001"],
    )

    model_config = {"json_schema_extra": {"example": _DUMMY_INPUT_EXAMPLE}}


class EvidenceSpanOut(BaseModel):
    quote: str = Field(description="환자 진술에서 그대로 발췌한 핵심 표현")
    interpretation: str = Field(description="이 표현이 분류 결정에 기여하는 방식")


class StageLogOut(BaseModel):
    stage: int = Field(description="분류 단계 (2 | 3 | 4)")
    selection: str = Field(description="선택된 분류 항목")
    confidence: str = Field(description="확신 수준: '높음' | '중간' | '낮음'")
    evidence_spans: list[EvidenceSpanOut] = Field(description="근거 발췌 목록")
    reason: str = Field(description="종합 선택 근거")


class TriageStateOut(BaseModel):
    session_id: str = Field(description="세션 식별자")
    stage2_selection: Optional[str] = Field(None, description="Stage 2 분류 결과 (주증상 계통)")
    stage3_selection: Optional[str] = Field(None, description="Stage 3 분류 결과 (세부 증상)")
    stage4_selection: Optional[str] = Field(None, description="Stage 4 분류 결과 (구체 임상 상태)")
    retriage_target: Optional[str] = Field(None, description="마지막 재평가 시작 단계")
    retriage_action: Optional[str] = Field(None, description="마지막 재평가 결정 ('추가 질문' | '재분류')")
    additional_questions: list[str] = Field(default_factory=list, description="현재 미답 추가 질문 목록")
    classification_log: list[StageLogOut] = Field(
        default_factory=list,
        description="stage별 분류 근거 누적 로그 (가장 최근 사이클 기준)",
    )

    model_config = {"json_schema_extra": {"example": _DUMMY_STATE_EXAMPLE}}


class InputResponse(BaseModel):
    session_id: str
    message: str = Field(description="처리 결과 요약 메시지")
    state: TriageStateOut


class ResetResponse(BaseModel):
    session_id: str
    message: str


# ── 앱 상태 (전역) ────────────────────────────────────────────────────────────
_graph = None
_memory: MemorySaver = None
_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 LLM + 그래프 초기화."""
    global _graph, _memory
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    _memory = MemorySaver()
    _graph = build_graph(llm, checkpointer=_memory)
    yield
    # 종료 시 정리 필요한 리소스 없음


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pre-KTAS API",
    description=(
        "## Pre-KTAS 실시간 중증도 분류 API\n\n"
        "STT 또는 키보드 입력을 받아 **LangGraph 기반 Multi-stage KTAS 분류**를 수행합니다.\n\n"
        "### 기본 사용 흐름\n"
        "1. `POST /triage/input` 으로 텍스트 전송 → 자동 분류 실행\n"
        "2. `GET /triage/state` 로 현재 분류 상태 및 근거(evidence_spans) 조회\n"
        "3. 새로운 정보 도착 시 `POST /triage/input` 재호출 → 재평가 수행\n"
        "4. 새 환자 시 `POST /triage/reset` 으로 세션 초기화\n\n"
        "### 더미 데이터 테스트\n"
        "`POST /triage/input` 의 **Try it out** → Execute 버튼으로 예시 데이터를 바로 실행할 수 있습니다."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _build_config(session_id: str) -> dict:
    """LangGraph invoke/stream 에 전달할 config 생성."""
    return {"configurable": {"thread_id": session_id}}


def _extract_state_out(session_id: str, raw: dict) -> TriageStateOut:
    """그래프 raw state 딕셔너리를 응답 모델로 변환."""
    raw_log = raw.get("classification_log") or []
    log_out = [
        StageLogOut(
            stage=entry["stage"],
            selection=entry["selection"],
            confidence=entry["confidence"],
            evidence_spans=[
                EvidenceSpanOut(**span) for span in entry.get("evidence_spans", [])
            ],
            reason=entry["reason"],
        )
        for entry in raw_log
    ]
    return TriageStateOut(
        session_id=session_id,
        stage2_selection=raw.get("stage2_selection"),
        stage3_selection=raw.get("stage3_selection"),
        stage4_selection=raw.get("stage4_selection"),
        retriage_target=raw.get("retriage_target"),
        retriage_action=raw.get("retriage_action"),
        additional_questions=raw.get("additional_questions") or [],
        classification_log=log_out,
    )


def _run_graph(session_id: str, user_input: str) -> dict:
    """그래프를 동기 실행하여 최종 state를 반환한다 (to_thread 에서 호출)."""
    config = _build_config(session_id)
    final_state: dict = {}
    for chunk in _graph.stream(
        {"user_input": user_input},
        config=config,
        stream_mode="values",
    ):
        final_state = chunk
    return final_state


def _summarize_message(state: dict) -> str:
    """처리 결과를 한 줄 요약 메시지로 변환."""
    action = state.get("retriage_action")
    s4 = state.get("stage4_selection")
    s3 = state.get("stage3_selection")
    s2 = state.get("stage2_selection")

    if action == "추가 질문":
        n = len(state.get("additional_questions") or [])
        return f"추가 질문 {n}개 생성됨"
    if s4:
        return f"stage4 분류 완료 → {s4}"
    if s3:
        return f"stage3 분류 완료 → {s3}"
    if s2:
        return f"stage2 분류 완료 → {s2}"
    return "그래프 실행 완료 (분류 결과 없음)"


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="헬스체크")
async def health_check():
    """서버 및 그래프 초기화 상태를 확인합니다."""
    return {"status": "ok", "graph_ready": _graph is not None}


@app.post(
    "/triage/input",
    response_model=InputResponse,
    tags=["Triage"],
    summary="텍스트 입력 → 분류 실행",
    responses={
        200: {
            "description": "분류 성공",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "patient_001",
                        "message": "stage4 분류 완료 → NSTEMI/불안정 협심증 의심",
                        "state": _DUMMY_STATE_EXAMPLE,
                    }
                }
            },
        },
        503: {"description": "그래프가 아직 초기화되지 않음"},
    },
)
async def submit_input(request: InputRequest):
    """
    STT 또는 키보드 텍스트를 수신하고 Pre-KTAS 분류 그래프를 실행합니다.

    - 같은 `session_id`로 반복 호출 시 **이전 분류 상태에 이어서** 재평가됩니다.
    - 분류 결과와 함께 각 단계별 **evidence_spans**(근거 발췌)가 반환됩니다.
    - 추가 정보가 필요한 경우 `additional_questions` 에 질문이 담겨 반환됩니다.

    ### 더미 데이터 예시
    아래 기본값을 그대로 Execute 하면 58세 남성 흉통 환자가 시뮬레이션됩니다.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="그래프가 초기화되지 않았습니다.")

    async with _lock:
        final_state = await asyncio.to_thread(
            _run_graph, request.session_id, request.text
        )

    state_out = _extract_state_out(request.session_id, final_state)
    return InputResponse(
        session_id=request.session_id,
        message=_summarize_message(final_state),
        state=state_out,
    )


@app.get(
    "/triage/state",
    response_model=TriageStateOut,
    tags=["Triage"],
    summary="현재 분류 상태 조회",
    responses={
        200: {
            "description": "현재 분류 상태",
            "content": {"application/json": {"example": _DUMMY_STATE_EXAMPLE}},
        },
        404: {"description": "해당 세션의 상태가 없음"},
    },
)
async def get_state(
    session_id: str = Query(default="default", description="조회할 세션 ID", examples=["patient_001"]),
):
    """
    지정한 세션의 현재 분류 상태를 조회합니다.

    - `classification_log`: stage 2→3→4 각 단계의 분류 결과와 근거 발췌 목록
    - `additional_questions`: 아직 미답된 추가 질문 목록
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="그래프가 초기화되지 않았습니다.")

    config = _build_config(session_id)
    snapshot = _graph.get_state(config)

    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"세션 '{session_id}' 의 상태가 없습니다.")

    return _extract_state_out(session_id, snapshot.values)


@app.post(
    "/triage/reset",
    response_model=ResetResponse,
    tags=["Triage"],
    summary="세션 초기화",
)
async def reset_session(
    session_id: str = Query(default="default", description="초기화할 세션 ID", examples=["patient_001"]),
):
    """
    지정한 세션의 분류 상태를 초기화합니다.

    새 환자가 내원하거나 처음부터 재분류가 필요한 경우 호출하세요.
    초기화 후 `POST /triage/input` 으로 새 텍스트를 제출하면 됩니다.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="그래프가 초기화되지 않았습니다.")

    config = _build_config(session_id)
    initial = get_initial_state()

    # MemorySaver에 초기 state를 덮어쓰기
    await asyncio.to_thread(_graph.update_state, config, initial)

    return ResetResponse(
        session_id=session_id,
        message=f"세션 '{session_id}' 초기화 완료",
    )
