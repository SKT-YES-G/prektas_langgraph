"""
Stage 2 Classifier 노드

역할:
  - 현재 stage2_candidates 안에서 가장 적합한 분류 1개를 선택한다.
  - 선택 후 해당 분류에 맞는 stage3_candidates 를 매핑하여 state에 저장한다.
  - 다음 노드: stage3_classifier (항상)
"""

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from pre_ktas.graph.state import GraphState
from nodes.data.ktas_candidates import STAGE2_CANDIDATES, get_stage3_candidates


# ── Structured Output Schema ──────────────────────────────────────────────────

class EvidenceSpan(BaseModel):
    quote: str = Field(
        description="환자 진술 또는 정보에서 그대로 발췌한 핵심 표현. "
                    "원문을 변형하지 말고 짧게(5단어 이내) 인용할 것."
    )
    interpretation: str = Field(
        description="이 표현이 분류 결정에 어떻게 기여하는지 한 줄로 설명. "
                    "예) '숨이 차요' → 호흡기계 증상을 직접 지시"
    )


class Stage2Classification(BaseModel):
    selection: str = Field(
        description="stage2_candidates 목록 중 가장 적합한 분류 항목 1개 (목록에 있는 문자열 그대로)."
    )
    confidence: str = Field(description="확신 수준: '높음' | '중간' | '낮음'")
    evidence_spans: list[EvidenceSpan] = Field(
        description="선택 근거가 되는 환자 진술 발췌 목록. 1~3개 추출."
    )
    reason: str = Field(description="선택 종합 근거 (1-2 문장)")


_SYSTEM_PROMPT = """\
당신은 Pre-KTAS 응급 분류 전문가입니다.

[환자 정보 및 진술]
{user_input}

[대화 히스토리 요약]
{history}

[Stage 2 분류 후보군]
{candidates}

위 후보군 중에서 현재 환자의 주증상 계통을 가장 잘 나타내는 항목 하나를 선택하세요.
반드시 후보군에 있는 문자열 그대로 선택해야 합니다.

evidence_spans에는 환자 진술에서 이 분류를 뒷받침하는 핵심 표현을 원문 그대로 발췌하세요.
예시) 진술: "어제부터 숨이 많이 차고 SpO2가 92%입니다"
  → quote: "숨이 많이 차고", interpretation: "호흡곤란 증상으로 호흡기계를 직접 지시"
  → quote: "SpO2가 92%",    interpretation: "산소포화도 저하로 호흡기계 중증도 근거"
"""


# ── Node Factory ──────────────────────────────────────────────────────────────

def make_stage2_classifier_node(llm):
    """LLM을 주입받아 stage2 classifier 노드 함수를 반환한다."""

    structured_llm = llm.with_structured_output(Stage2Classification)

    def stage2_classifier_node(state: GraphState) -> dict:
        user_input = state.get("user_input", "")
        candidates = state.get("stage2_candidates") or STAGE2_CANDIDATES
        candidates_str = "\n".join(f"- {c}" for c in candidates)

        # 메시지 히스토리에서 간단한 맥락 추출
        messages_history = state.get("messages") or []
        history_text = _summarize_history(messages_history)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(
                user_input=user_input,
                history=history_text,
                candidates=candidates_str,
            )),
            HumanMessage(content=user_input),
        ]

        result: Stage2Classification = structured_llm.invoke(messages)

        # 후보군에 없는 값이 반환되면 첫 번째 후보를 fallback으로 사용
        selection = result.selection if result.selection in candidates else candidates[0]

        # stage3 후보군 매핑
        stage3_candidates = get_stage3_candidates(selection)

        # 분류 로그 항목 생성
        log_entry = {
            "stage": 2,
            "selection": selection,
            "confidence": result.confidence,
            "evidence_spans": [e.model_dump() for e in result.evidence_spans],
            "reason": result.reason,
        }
        existing_log = state.get("classification_log") or []

        return {
            "stage2_selection": selection,
            "stage3_candidates": stage3_candidates,
            "current_stage": 2,
            "classification_log": existing_log + [log_entry],
        }

    return stage2_classifier_node


def _summarize_history(messages: list) -> str:
    """메시지 히스토리를 간단한 텍스트로 변환."""
    if not messages:
        return "(이전 대화 없음)"
    lines = []
    for msg in messages[-6:]:  # 최근 6개만
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            lines.append(f"[{role}] {content[:100]}")
    return "\n".join(lines) if lines else "(이전 대화 없음)"
