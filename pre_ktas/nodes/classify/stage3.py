"""
Stage 3 Classifier 노드

역할:
  - stage3_candidates (stage2 선택 결과로 매핑된 후보군) 안에서
    가장 적합한 세부 증상 1개를 선택한다.
  - 선택 후 해당 분류에 맞는 stage4_candidates 를 매핑하여 state에 저장한다.
  - 다음 노드: stage4_classifier (항상)
"""

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from pre_ktas.graph.state import GraphState
from pre_ktas.nodes.data.ktas_candidates import get_stage4_candidates


# ── Structured Output Schema ──────────────────────────────────────────────────

class EvidenceSpan(BaseModel):
    quote: str = Field(
        description="환자 진술 또는 정보에서 그대로 발췌한 핵심 표현. "
                    "원문을 변형하지 말고 짧게(5단어 이내) 인용할 것."
    )
    interpretation: str = Field(
        description="이 표현이 세부 증상 분류 결정에 어떻게 기여하는지 한 줄로 설명."
    )


class Stage3Classification(BaseModel):
    selection: str = Field(
        description="stage3_candidates 목록 중 가장 적합한 세부 증상 항목 1개 (목록에 있는 문자열 그대로)."
    )
    confidence: str = Field(description="확신 수준: '높음' | '중간' | '낮음'")
    evidence_spans: list[EvidenceSpan] = Field(
        description="선택 근거가 되는 환자 진술 발췌 목록. 1~3개 추출."
    )
    reason: str = Field(description="선택 종합 근거 (1-2 문장)")


_SYSTEM_PROMPT = """\
당신은 Pre-KTAS 응급 분류 전문가입니다.

[Stage 2 확정 분류]
{stage2}

[환자 정보 및 진술]
{user_input}

[Stage 3 분류 후보군 (Stage 2 '{stage2}' 의 세부 증상)]
{candidates}

위 후보군 중에서 현재 환자의 세부 증상을 가장 잘 나타내는 항목 하나를 선택하세요.
반드시 후보군에 있는 문자열 그대로 선택해야 합니다.

evidence_spans에는 환자 진술에서 이 세부 증상 분류를 뒷받침하는 핵심 표현을 원문 그대로 발췌하세요.
예시) 진술: "왼쪽 가슴이 쥐어짜는 듯이 아프고 등으로 뻗쳐요"
  → quote: "왼쪽 가슴이 쥐어짜는",    interpretation: "전형적인 심장성 흉통 양상"
  → quote: "등으로 뻗쳐요",            interpretation: "방사통으로 심근경색 가능성 시사"
"""


# ── Node Factory ──────────────────────────────────────────────────────────────

def make_stage3_classifier_node(llm):
    """LLM을 주입받아 stage3 classifier 노드 함수를 반환한다."""

    structured_llm = llm.with_structured_output(Stage3Classification)

    def stage3_classifier_node(state: GraphState) -> dict:
        user_input = state.get("user_input", "")
        stage2 = state.get("stage2_selection") or "미분류"
        candidates = state.get("stage3_candidates") or []

        if not candidates:
            # 후보군이 비어있으면 분류 불가 → 빈 결과 반환
            return {"stage3_selection": None, "stage4_candidates": []}

        candidates_str = "\n".join(f"- {c}" for c in candidates)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(
                stage2=stage2,
                user_input=user_input,
                candidates=candidates_str,
            )),
            HumanMessage(content=user_input),
        ]

        result: Stage3Classification = structured_llm.invoke(messages)

        selection = result.selection if result.selection in candidates else candidates[0]

        # stage4 후보군 매핑
        stage4_candidates = get_stage4_candidates(selection)

        log_entry = {
            "stage": 3,
            "selection": selection,
            "confidence": result.confidence,
            "evidence_spans": [e.model_dump() for e in result.evidence_spans],
            "reason": result.reason,
        }
        existing_log = state.get("classification_log") or []

        return {
            "stage3_selection": selection,
            "stage4_candidates": stage4_candidates,
            "current_stage": 3,
            "classification_log": existing_log + [log_entry],
        }

    return stage3_classifier_node
