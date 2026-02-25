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
        description="stage4_candidates 목록 중 가장 적합한 구체 상태 항목 1개 (목록에 있는 문자열 그대로). "
                    "confidence='낮음'이면 빈 문자열 가능.",
    )
    confidence: str = Field(description="확신 수준: '높음' | '중간' | '낮음'")
    questions: list[str] = Field(
        default_factory=list,
        description="confidence='낮음'일 때, 분류에 필요하지만 환자 진술에서 확인할 수 없었던 "
                    "임상 정보를 묻는 질문 1~3개. confidence='높음'|'중간'이면 빈 리스트.",
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="선택 근거가 되는 환자 진술·활력징후 발췌 목록. 1~3개 추출.",
    )
    reason: str = Field(default="", description="선택 종합 근거 (1-2 문장)")


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

★ Stage 4는 임상 수치가 분류를 결정합니다 ★
후보군 목록을 먼저 살피고, 후보에 사용된 키워드(통증 점수, 급성/만성, 중증/중등도/경증 등)에
해당하는 임상 데이터를 환자 진술에서 찾아 매칭하세요.

후보군에 자주 등장하는 구분 축과 판단 방법:
┌─────────────────┬───────────────────────────────────────────┐
│ 후보 키워드      │ 환자 진술에서 찾아야 할 것                 │
├─────────────────┼───────────────────────────────────────────┤
│ (8-10)/(4-7)/(<4)│ 통증 점수 NRS·VAS 또는 표현 강도를 환산  │
│                  │  극심/참을수없다→8-10, 중간/꽤→4-7,       │
│                  │  약간/견딜만하다→<4                        │
│ 급성 / 만성      │ 발생 시점: 수시간~수일=급성, 수주이상=만성 │
│ 중증/중등도/경증  │ 호흡곤란·출혈·탈수 등의 심각도 표현       │
│ 쇼크             │ 수축기혈압<90, 맥박>120, 의식저하 등      │
│ 혈역학적 장애     │ 비정상 혈압·맥박이지만 쇼크까진 아닌 상태  │
│ 의식변화/무의식   │ GCS 점수 또는 의식 상태 기술              │
│ 열/면역저하       │ 체온 ≥38℃ 여부, 면역저하 기저질환 유무   │
│ 패혈증/SIRS      │ SIRS 기준 충족 개수 (2개 vs 3개 이상)     │
└─────────────────┴───────────────────────────────────────────┘

환자 진술에 해당 수치가 없고 표현만으로도 판단이 어려우면:
- confidence를 '낮음'으로 설정하고
- questions에 부족한 임상 정보를 묻는 질문을 1~3개 포함하세요.
- 추정으로 분류하지 마세요. 잘못된 분류보다 재질문이 낫습니다.

confidence가 '높음' 또는 '중간'이면 evidence_spans에 판단에 사용한 수치·표현을 원문 그대로 발췌하세요.
예시) 진술: "어제부터 두통이 너무 심해서 참을 수가 없어요"
  → quote: "참을 수가 없어요",  interpretation: "NRS 8-10 수준의 극심한 통증"
  → quote: "어제부터",          interpretation: "급성 발생 (수일 이내)"
예시) 진술: "혈압 85/60이고 어지럽고 의식이 흐려져요"
  → quote: "혈압 85/60",        interpretation: "수축기혈압 <90 으로 쇼크 기준 충족"
  → quote: "의식이 흐려져요",   interpretation: "의식변화 동반, 쇼크 중증도 시사"
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

        # ── confidence '낮음' + 질문이 있으면 → 분류하지 않고 재질문 ──
        if result.confidence == "낮음" and result.questions:
            existing_q = state.get("additional_questions") or []
            return {
                "retriage_action": "추가 질문",
                "additional_questions": existing_q + result.questions,
                "current_stage": 4,
            }

        # ── 분류 확정 ──
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
