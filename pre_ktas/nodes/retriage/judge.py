"""
재평가 판단 노드 (Retriage Judge Node)

역할: N초마다 들어오는 스트리밍/키보드 입력을 받아
      현재 분류 상태를 검토하고,
      어느 단계(stage2 / stage3 / stage4)부터 재평가할지 결정한다.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from pre_ktas.graph.state import GraphState


# ── Structured Output Schema ──────────────────────────────────────────────────

class RetraigeDecision(BaseModel):
    target: Literal["stage2", "stage3", "stage4"] = Field(
        description=(
            "재평가를 시작할 분류 단계. "
            "주증상 계통이 바뀔 가능성이 있으면 stage2, "
            "계통은 같지만 세부 증상이 달라지면 stage3, "
            "중증도·세부 상태만 변하면 stage4."
        )
    )
    reason: str = Field(description="판단 근거 (1-2 문장)")


_SYSTEM_PROMPT = """\
당신은 Pre-KTAS(한국형 중증도 분류) 재평가 전문가입니다.

[현재 분류 상태]
- Stage 2 (주증상 계통): {stage2}
- Stage 3 (세부 증상): {stage3}
- Stage 4 (구체 상태): {stage4}

[새로운 정보]
{user_input}

위 새로운 정보를 바탕으로, 어느 단계부터 재평가해야 하는지 판단하세요.

판단 기준:
- stage2: 주증상 계통 자체가 변경될 가능성이 있을 때
           예) 호흡기계 → 심혈관계
- stage3: 계통은 동일하지만 세부 증상이 다를 때
           예) 호흡곤란 → 가슴통증(호흡기성)
- stage4: 중증도 또는 세부 상태만 변경될 때
           예) 경증 호흡곤란 → 중증 호흡곤란
"""


# ── Node Factory ──────────────────────────────────────────────────────────────

def make_retriage_judge_node(llm):
    """LLM을 주입받아 재평가 판단 노드 함수를 반환한다."""

    structured_llm = llm.with_structured_output(RetraigeDecision)

    def retriage_judge_node(state: GraphState) -> dict:
        user_input = state.get("user_input", "")
        stage2 = state.get("stage2_selection") or "미분류"
        stage3 = state.get("stage3_selection") or "미분류"
        stage4 = state.get("stage4_selection") or "미분류"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(
                stage2=stage2,
                stage3=stage3,
                stage4=stage4,
                user_input=user_input,
            )),
            HumanMessage(content=user_input),
        ]

        decision: RetraigeDecision = structured_llm.invoke(messages)

        return {
            "retriage_target": decision.target,
            # retriage_action은 이 노드에서 초기화하지 않음
            # (이전 사이클 값이 남아 있으면 오동작하므로 명시적 초기화)
            "retriage_action": None,
        }

    return retriage_judge_node
