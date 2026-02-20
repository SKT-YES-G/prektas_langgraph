"""
Stage 4 Classifier 노드

역할:
  - stage4_candidates (stage3 선택 결과로 매핑된 후보군) 안에서
    가장 적합한 구체적 임상 상태 1개를 선택한다.
  - 선택 결과가 최종 분류 state로 저장되며, 그래프 흐름이 종료된다.
"""

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from pre_ktas.graph.state import GraphState
from pre_ktas.nodes.data.ktas_candidates import get_ktas_level


# ── Structured Output Schema ──────────────────────────────────────────────────

class EvidenceSpan(BaseModel):
    quote: str = Field(
        description="환자 진술 또는 활력징후 수치에서 그대로 발췌한 핵심 표현. "
                    "원문을 변형하지 말고 짧게(5단어 이내) 인용할 것."
    )
    interpretation: str = Field(
        description="이 표현이 구체적 임상 상태 분류 결정에 어떻게 기여하는지 한 줄로 설명."
    )


class Stage4Classification(BaseModel):
    selection: str = Field(
        description="stage4_candidates 목록 중 가장 적합한 구체 상태 항목 1개 (목록에 있는 문자열 그대로)."
    )
    confidence: str = Field(description="확신 수준: '높음' | '중간' | '낮음'")
    evidence_spans: list[EvidenceSpan] = Field(
        description="선택 근거가 되는 환자 진술·활력징후 발췌 목록. 1~3개 추출."
    )
    reason: str = Field(description="선택 종합 근거 (1-2 문장)")


_SYSTEM_PROMPT = """\
당신은 Pre-KTAS 응급 분류 전문가입니다.

[현재 분류 상태]
- Stage 2: {stage2}
- Stage 3: {stage3}

[환자 정보 및 진술]
{user_input}

[Stage 4 분류 후보군 (Stage 3 '{stage3}' 의 구체적 임상 상태)]
{candidates}

위 후보군 중에서 현재 환자의 구체적인 임상 상태를 가장 잘 나타내는 항목 하나를 선택하세요.
반드시 후보군에 있는 문자열 그대로 선택해야 합니다.

주요 판단 기준:
- 활력 징후(혈압, 맥박, 호흡수, 체온, SpO2)의 이상 여부
- 의식 수준 변화 여부
- 증상 발생 시각 및 진행 속도
- 동반 증상의 심각도

evidence_spans에는 임상 상태를 직접적으로 뒷받침하는 수치나 진술을 원문 그대로 발췌하세요.
예시) 진술: "혈압 190/110이고 갑자기 두통이 심해지며 말이 어눌해졌어요"
  → quote: "혈압 190/110",         interpretation: "고혈압 응급 수준, 장기손상 동반 가능성"
  → quote: "말이 어눌해졌어요",    interpretation: "신경학적 증상으로 장기손상(뇌) 시사"
"""


# ── Node Factory ──────────────────────────────────────────────────────────────

def make_stage4_classifier_node(llm):
    """LLM을 주입받아 stage4 classifier 노드 함수를 반환한다."""

    structured_llm = llm.with_structured_output(Stage4Classification)

    def stage4_classifier_node(state: GraphState) -> dict:
        user_input = state.get("user_input", "")
        stage2 = state.get("stage2_selection") or "미분류"
        stage3 = state.get("stage3_selection") or "미분류"
        candidates = state.get("stage4_candidates") or []

        if not candidates:
            return {"stage4_selection": None, "current_stage": 4}

        candidates_str = "\n".join(f"- {c}" for c in candidates)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(
                stage2=stage2,
                stage3=stage3,
                user_input=user_input,
                candidates=candidates_str,
            )),
            HumanMessage(content=user_input),
        ]

        result: Stage4Classification = structured_llm.invoke(messages)

        selection = result.selection if result.selection in candidates else candidates[0]

        log_entry = {
            "stage": 4,
            "selection": selection,
            "confidence": result.confidence,
            "evidence_spans": [e.model_dump() for e in result.evidence_spans],
            "reason": result.reason,
        }
        existing_log = state.get("classification_log") or []

        # CSV 기반 KTAS 등급 매핑 (stage2 + stage3 + stage4 → 1~5)
        ktas_level = get_ktas_level(stage2, stage3, selection)

        return {
            "stage4_selection": selection,
            "current_stage": 4,
            "classification_log": existing_log + [log_entry],
            "final_ktas_level": ktas_level,
        }

    return stage4_classifier_node
