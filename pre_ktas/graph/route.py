"""
라우팅 함수 모음

각 노드의 conditional_edge 에서 호출되는 함수들.
state 를 읽어 다음 노드 이름(str)을 반환한다.
"""

from pre_ktas.graph.state import GraphState


# ──────────────────────────────────────────────────────────────────────────────
# 1. 재평가 판단 노드 → stage-n 재평가 노드
# ──────────────────────────────────────────────────────────────────────────────

def route_retriage_target(state: GraphState) -> str:
    """retriage_judge 가 설정한 retriage_target 에 따라 라우팅.

    반환값이 add_conditional_edges 의 path_map 키와 일치해야 한다.
    """
    target = state.get("retriage_target", "stage4")
    return f"retriage_{target}"   # "retriage_stage2" | "retriage_stage3" | "retriage_stage4"


# ──────────────────────────────────────────────────────────────────────────────
# 2. stage-n 재평가 노드 → (추가 질문 | stage-n classifier)
# ──────────────────────────────────────────────────────────────────────────────

def route_retriage_stage2_action(state: GraphState) -> str:
    """retriage_stage2 이후 라우팅.

    - '추가 질문' → ask_question
    - '재분류'    → stage2_classifier
    """
    action = state.get("retriage_action")
    if action == "추가 질문":
        return "ask_question"
    return "stage2_classifier"   # 기본: 재분류


def route_retriage_stage3_action(state: GraphState) -> str:
    """retriage_stage3 이후 라우팅.

    - '추가 질문' → ask_question
    - '재분류'    → stage3_classifier
    """
    action = state.get("retriage_action")
    if action == "추가 질문":
        return "ask_question"
    return "stage3_classifier"


def route_retriage_stage4_action(state: GraphState) -> str:
    """retriage_stage4 이후 라우팅.

    - '추가 질문' → ask_question
    - '재분류'    → stage4_classifier
    """
    action = state.get("retriage_action")
    if action == "추가 질문":
        return "ask_question"
    return "stage4_classifier"
