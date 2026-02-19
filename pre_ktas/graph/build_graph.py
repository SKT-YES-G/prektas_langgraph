"""
그래프 빌더 (build_graph)

노드 등록 → 엣지 연결 → 컴파일 순서로 그래프를 구성한다.

전체 흐름:
  [START]
    │
    ▼
  retriage_judge  ─── retriage_target ─┬─▶ retriage_stage2 ─── action ─┬─▶ ask_question ──▶ [END]
                                        │                                └─▶ stage2_classifier ─▶ stage3_classifier ─▶ stage4_classifier ─▶ [END]
                                        ├─▶ retriage_stage3 ─── action ─┬─▶ ask_question ──▶ [END]
                                        │                                └─▶ stage3_classifier ─▶ stage4_classifier ─▶ [END]
                                        └─▶ retriage_stage4 ─── action ─┬─▶ ask_question ──▶ [END]
                                                                         └─▶ stage4_classifier ─▶ [END]
"""

from langgraph.graph import StateGraph, END

from pre_ktas.graph.state import GraphState
from pre_ktas.graph.route import (
    route_retriage_target,
    route_retriage_stage2_action,
    route_retriage_stage3_action,
    route_retriage_stage4_action,
)
from nodes.data.ktas_candidates import STAGE2_CANDIDATES
from nodes.retriage.judge import make_retriage_judge_node
from nodes.retriage.stage2 import make_retriage_stage2_node
from nodes.retriage.stage3 import make_retriage_stage3_node
from nodes.retriage.stage4 import make_retriage_stage4_node
from nodes.classify.stage2 import make_stage2_classifier_node
from nodes.classify.stage3 import make_stage3_classifier_node
from nodes.classify.stage4 import make_stage4_classifier_node
from nodes.ask.question import ask_question_node


def build_graph(llm, checkpointer=None):
    """
    LLM 인스턴스를 받아 Pre-KTAS 재평가 그래프를 컴파일하여 반환한다.

    Args:
        llm: langchain BaseChatModel (예: ChatOpenAI, ChatAnthropic 등)
        checkpointer: LangGraph checkpointer (예: MemorySaver).
                      None 이면 체크포인트 없이 컴파일된다.
                      FastAPI 처럼 호출 간 상태를 유지할 때 MemorySaver를 전달한다.

    Returns:
        CompiledGraph: invoke / stream 가능한 LangGraph 그래프
    """

    # ── 노드 함수 생성 (LLM 주입) ─────────────────────────────────────────
    retriage_judge     = make_retriage_judge_node(llm)
    retriage_stage2    = make_retriage_stage2_node(llm)
    retriage_stage3    = make_retriage_stage3_node(llm)
    retriage_stage4    = make_retriage_stage4_node(llm)
    stage2_classifier  = make_stage2_classifier_node(llm)
    stage3_classifier  = make_stage3_classifier_node(llm)
    stage4_classifier  = make_stage4_classifier_node(llm)

    # ── 그래프 초기화 ─────────────────────────────────────────────────────
    graph = StateGraph(GraphState)

    # ── 노드 등록 ─────────────────────────────────────────────────────────
    graph.add_node("retriage_judge",    retriage_judge)
    graph.add_node("retriage_stage2",   retriage_stage2)
    graph.add_node("retriage_stage3",   retriage_stage3)
    graph.add_node("retriage_stage4",   retriage_stage4)
    graph.add_node("stage2_classifier", stage2_classifier)
    graph.add_node("stage3_classifier", stage3_classifier)
    graph.add_node("stage4_classifier", stage4_classifier)
    graph.add_node("ask_question",      ask_question_node)

    # ── 엔트리 포인트 ─────────────────────────────────────────────────────
    graph.set_entry_point("retriage_judge")

    # ── 재평가 판단 → stage-n 재평가 (conditional) ────────────────────────
    graph.add_conditional_edges(
        "retriage_judge",
        route_retriage_target,
        {
            "retriage_stage2": "retriage_stage2",
            "retriage_stage3": "retriage_stage3",
            "retriage_stage4": "retriage_stage4",
        },
    )

    # ── stage2 재평가 → (추가 질문 | stage2 classifier) ──────────────────
    graph.add_conditional_edges(
        "retriage_stage2",
        route_retriage_stage2_action,
        {
            "ask_question":      "ask_question",
            "stage2_classifier": "stage2_classifier",
        },
    )

    # ── stage3 재평가 → (추가 질문 | stage3 classifier) ──────────────────
    graph.add_conditional_edges(
        "retriage_stage3",
        route_retriage_stage3_action,
        {
            "ask_question":      "ask_question",
            "stage3_classifier": "stage3_classifier",
        },
    )

    # ── stage4 재평가 → (추가 질문 | stage4 classifier) ──────────────────
    graph.add_conditional_edges(
        "retriage_stage4",
        route_retriage_stage4_action,
        {
            "ask_question":      "ask_question",
            "stage4_classifier": "stage4_classifier",
        },
    )

    # ── Classifier cascade: stage2 → stage3 → stage4 → END ──────────────
    graph.add_edge("stage2_classifier", "stage3_classifier")
    graph.add_edge("stage3_classifier", "stage4_classifier")
    graph.add_edge("stage4_classifier", END)

    # ── 추가 질문 후 → END (다음 스트림 사이클 대기) ─────────────────────
    graph.add_edge("ask_question", END)

    return graph.compile(checkpointer=checkpointer)


def get_initial_state() -> dict:
    """그래프 최초 실행 시 사용할 초기 state 를 반환한다."""
    return {
        "messages":            [],
        "user_input":          "",
        "current_stage":       None,
        "stage2_selection":    None,
        "stage3_selection":    None,
        "stage4_selection":    None,
        "stage2_candidates":   STAGE2_CANDIDATES,
        "stage3_candidates":   [],
        "stage4_candidates":   [],
        "retriage_target":     None,
        "retriage_action":     None,
        "additional_questions": [],
        "patient_info":        {},
        "classification_log":  [],
        "final_ktas_level":    None,
    }
