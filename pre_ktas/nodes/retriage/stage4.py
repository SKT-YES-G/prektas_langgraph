"""
Stage 4 재평가 노드 (Retriage Stage 4)

역할: 재평가 판단 노드가 'stage4' 로 라우팅했을 때 실행된다.
      Stage 2, Stage 3 모두 확정된 상태에서 구체 상태(Stage 4)만 재검토한다.
        - '추가 질문' : 정보 부족 → 질문 생성 후 종료
        - '재분류'    : 충분한 정보 → stage4 classifier 로 진행
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from pre_ktas.graph.state import GraphState
from pre_ktas.nodes.data.ktas_candidates import get_stage4_candidates


# ── Structured Output Schema ──────────────────────────────────────────────────

class RetraigeStage4Decision(BaseModel):
    action: Literal["추가 질문", "재분류"] = Field(
        description=(
            "'추가 질문': 구체적인 상태를 결정하기에 정보가 부족. "
            "'재분류': 충분한 정보로 stage4 분류를 수행할 수 있음."
        )
    )
    questions: list[str] = Field(
        default_factory=list,
        description="action='추가 질문'일 때 질문 목록 (1~3개, 반드시 1개 이상). "
                    "action='재분류'이면 빈 리스트.",
    )
    reason: str = Field(description="결정 근거 (1-2 문장)")


_SYSTEM_PROMPT = """\
당신은 Pre-KTAS 응급 분류 전문가입니다.
Stage 2, Stage 3 분류는 확정되었고, Stage 4(구체적인 임상 상태)를 재검토합니다.

[현재 분류 상태]
- Stage 2: {stage2}
- Stage 3: {stage3}
- Stage 4 (현재): {stage4}

[Stage 4 후보군 (Stage 3 '{stage3}' 에 해당)]
{candidates}

[새로운 정보 / 환자 진술]
{user_input}

★ Stage 4 분류는 임상 데이터에 의해 결정됩니다 ★
후보군 목록을 살피세요. 후보에 통증 점수 구간(8-10/4-7/<4), 급성/만성,
중증/중등도/경증, 쇼크, 혈역학적 장애, 의식 수준, 열, 패혈증/SIRS 등의
키워드가 있다면, 해당 정보가 환자 진술에 있는지 확인하세요.

아래 중 하나를 결정하세요:
1) '추가 질문': 후보군을 구분하는 데 필요한 임상 정보가 부족합니다.
   **반드시 questions에 1~3개의 질문을 포함하세요. 빈 리스트는 허용되지 않습니다.**
   부족한 정보를 구체적으로 물어보세요. 예:
   - 통증 점수 구간이 필요하면 → "통증 정도를 0-10 점으로 표현하면 몇 점인가요?"
   - 급성/만성 구분이 필요하면 → "이 증상이 언제 처음 시작되었나요?"
   - 활력징후가 필요하면 → "현재 혈압, 맥박, 체온, SpO2 수치를 알려주세요."
   - 의식 수준이 필요하면 → "환자의 의식 상태는 어떤가요? (명료/혼미/반혼수/혼수)"
2) '재분류': 후보군을 구분할 수 있는 충분한 임상 정보가 있습니다.
"""


# ── Node Factory ──────────────────────────────────────────────────────────────

def make_retriage_stage4_node(llm):
    """LLM을 주입받아 stage4 재평가 노드 함수를 반환한다."""

    structured_llm = llm.with_structured_output(RetraigeStage4Decision)

    def retriage_stage4_node(state: GraphState) -> dict:
        user_input = state.get("user_input", "")
        stage2 = state.get("stage2_selection") or "미분류"
        stage3 = state.get("stage3_selection") or "미분류"
        stage4 = state.get("stage4_selection") or "미분류"
        candidates = get_stage4_candidates(stage3)
        candidates_str = "\n".join(f"- {c}" for c in candidates) if candidates else "(후보 없음)"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(
                stage2=stage2,
                stage3=stage3,
                stage4=stage4,
                candidates=candidates_str,
                user_input=user_input,
            )),
            HumanMessage(content=user_input),
        ]

        decision: RetraigeStage4Decision = structured_llm.invoke(messages)

        update: dict = {
            "retriage_action": decision.action,
            # stage4 재평가이므로 stage4 candidates를 state에 갱신
            "stage4_candidates": candidates,
        }

        if decision.action == "추가 질문":
            questions = decision.questions or [
                "현재 활력징후(혈압, 맥박, 체온, SpO2)를 알려주세요.",
                "증상의 중증도가 어느 정도인지 설명해 주세요.",
            ]
            existing = state.get("additional_questions") or []
            update["additional_questions"] = existing + questions

        return update

    return retriage_stage4_node
