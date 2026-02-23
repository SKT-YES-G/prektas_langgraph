"""
Stage 2 재평가 노드 (Retriage Stage 2)

역할: 재평가 판단 노드가 'stage2' 로 라우팅했을 때 실행된다.
      새 정보를 바탕으로 다음 중 하나를 결정한다:
        - '추가 질문' : 분류에 필요한 정보가 부족 → 질문 생성 후 종료
        - '재분류'    : 충분한 정보가 있음 → stage2 classifier 로 진행
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from pre_ktas.graph.state import GraphState
from pre_ktas.nodes.data.ktas_candidates import STAGE2_CANDIDATES


# ── Structured Output Schema ──────────────────────────────────────────────────

class RetraigeStage2Decision(BaseModel):
    action: Literal["추가 질문", "재분류"] = Field(
        description=(
            "'추가 질문': 분류를 확정하기에 정보가 부족해 질문을 생성해야 함. "
            "'재분류': 충분한 정보로 stage2 분류를 새로 수행할 수 있음."
        )
    )
    questions: list[str] = Field(
        default_factory=list,
        description="action='추가 질문'일 때 환자/보호자에게 물어볼 질문 목록 (1~3개, 반드시 1개 이상). "
                    "action='재분류'이면 빈 리스트.",
    )
    reason: str = Field(description="결정 근거 (1-2 문장)")


_SYSTEM_PROMPT = """\
당신은 Pre-KTAS 응급 분류 전문가입니다.
현재 Stage 2(주증상 계통) 분류를 재검토해야 합니다.

[현재 Stage 2 분류]
{stage2}

[Stage 2 전체 후보군]
{candidates}

[새로운 정보 / 환자 진술]
{user_input}

새로운 정보를 바탕으로 아래 중 하나를 결정하세요:
1) '추가 질문': Stage 2 계통을 결정하기에 정보가 충분하지 않아, \
질문이 필요합니다. **반드시 questions에 1~3개의 구체적인 질문을 포함하세요. 빈 리스트는 허용되지 않습니다.**
2) '재분류': 정보가 충분하여 Stage 2 후보군 중에서 바로 분류할 수 있습니다.
"""


# ── Node Factory ──────────────────────────────────────────────────────────────

def make_retriage_stage2_node(llm):
    """LLM을 주입받아 stage2 재평가 노드 함수를 반환한다."""

    structured_llm = llm.with_structured_output(RetraigeStage2Decision)

    def retriage_stage2_node(state: GraphState) -> dict:
        user_input = state.get("user_input", "")
        stage2 = state.get("stage2_selection") or "미분류"
        candidates_str = "\n".join(f"- {c}" for c in STAGE2_CANDIDATES)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(
                stage2=stage2,
                candidates=candidates_str,
                user_input=user_input,
            )),
            HumanMessage(content=user_input),
        ]

        decision: RetraigeStage2Decision = structured_llm.invoke(messages)

        update: dict = {"retriage_action": decision.action}

        if decision.action == "추가 질문":
            questions = decision.questions or [
                "주요 증상이 무엇인지 구체적으로 설명해 주세요.",
                "증상이 언제부터 시작되었나요?",
            ]
            existing = state.get("additional_questions") or []
            update["additional_questions"] = existing + questions

        return update

    return retriage_stage2_node
