"""
Stage 3 재평가 노드 (Retriage Stage 3)

역할: 재평가 판단 노드가 'stage3' 로 라우팅했을 때 실행된다.
      Stage 2 계통은 확정된 상태에서 세부 증상(Stage 3)만 재검토한다.
        - '추가 질문' : 정보 부족 → 질문 생성 후 종료
        - '재분류'    : 충분한 정보 → stage3 classifier 로 진행
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from pre_ktas.graph.state import GraphState
from pre_ktas.nodes.data.ktas_candidates import get_stage3_candidates


# ── Structured Output Schema ──────────────────────────────────────────────────

class RetraigeStage3Decision(BaseModel):
    action: Literal["추가 질문", "재분류"] = Field(
        description=(
            "'추가 질문': 세부 증상을 결정하기에 정보가 부족. "
            "'재분류': 충분한 정보로 stage3 분류를 수행할 수 있음."
        )
    )
    questions: list[str] = Field(
        default_factory=list,
        description="action='추가 질문'일 때 질문 목록 (최대 3개). "
                    "action='재분류'이면 빈 리스트.",
    )
    reason: str = Field(description="결정 근거 (1-2 문장)")


_SYSTEM_PROMPT = """\
당신은 Pre-KTAS 응급 분류 전문가입니다.
Stage 2 분류는 확정되었고, Stage 3(세부 증상)를 재검토합니다.

[현재 Stage 2 분류]
{stage2}

[현재 Stage 3 분류]
{stage3}

[Stage 3 후보군 (Stage 2 '{stage2}' 에 해당)]
{candidates}

[새로운 정보 / 환자 진술]
{user_input}

새로운 정보를 바탕으로 아래 중 하나를 결정하세요:
1) '추가 질문': Stage 3 세부 증상을 결정하기에 정보가 충분하지 않아, \
질문이 필요합니다. 최대 3개의 구체적인 질문을 생성하세요.
2) '재분류': 정보가 충분하여 Stage 3 후보군 중에서 바로 분류할 수 있습니다.
"""


# ── Node Factory ──────────────────────────────────────────────────────────────

def make_retriage_stage3_node(llm):
    """LLM을 주입받아 stage3 재평가 노드 함수를 반환한다."""

    structured_llm = llm.with_structured_output(RetraigeStage3Decision)

    def retriage_stage3_node(state: GraphState) -> dict:
        user_input = state.get("user_input", "")
        stage2 = state.get("stage2_selection") or "미분류"
        stage3 = state.get("stage3_selection") or "미분류"
        candidates = get_stage3_candidates(stage2)
        candidates_str = "\n".join(f"- {c}" for c in candidates) if candidates else "(후보 없음)"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(
                stage2=stage2,
                stage3=stage3,
                candidates=candidates_str,
                user_input=user_input,
            )),
            HumanMessage(content=user_input),
        ]

        decision: RetraigeStage3Decision = structured_llm.invoke(messages)

        update: dict = {
            "retriage_action": decision.action,
            # stage3 재평가이므로 stage3 candidates를 state에 갱신
            "stage3_candidates": candidates,
        }

        if decision.action == "추가 질문" and decision.questions:
            existing = state.get("additional_questions") or []
            update["additional_questions"] = existing + decision.questions

        return update

    return retriage_stage3_node
