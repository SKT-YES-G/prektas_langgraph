from typing import TypedDict, Optional, List, Literal, Annotated
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    """Pre-KTAS LangGraph 상태 정의.

    스트리밍 입력과 키보드 입력을 모두 수용하며,
    KTAS 분류 단계별 선택 결과와 재평가 흐름을 추적합니다.
    """

    # ── 대화 히스토리 ─────────────────────────────────────
    # add_messages reducer: 기존 리스트에 append (덮어쓰기 아님)
    messages: Annotated[list, add_messages]

    # ── 사용자 입력 ───────────────────────────────────────
    # 스트리밍(음성 전사 등) 또는 키보드 입력을 공통 필드로 수신
    user_input: str

    # ── 분류 단계별 선택 결과 ─────────────────────────────
    current_stage: Optional[int]          # 현재 진행 중인 단계 (2 | 3 | 4)
    stage2_selection: Optional[str]       # 선택된 stage-2 분류
    stage3_selection: Optional[str]       # 선택된 stage-3 분류
    stage4_selection: Optional[str]       # 선택된 stage-4 분류

    # ── 단계별 후보군 ─────────────────────────────────────
    # 초기화 시 stage2_candidates를 STAGE2_CANDIDATES로 세팅
    stage2_candidates: List[str]
    stage3_candidates: List[str]          # stage2 선택 후 매핑으로 결정
    stage4_candidates: List[str]          # stage3 선택 후 매핑으로 결정

    # ── 재평가 흐름 ──────────────────────────────────────
    retriage_target: Optional[Literal["stage2", "stage3", "stage4"]]
    # 재평가 판단 노드가 설정 → 해당 재평가 노드로 라우팅

    retriage_action: Optional[Literal["추가 질문", "재분류"]]
    # 재평가 노드가 설정 → 추가 질문 / 재분류(classifier)

    # ── 추가 질문 ─────────────────────────────────────────
    additional_questions: List[str]       # 재평가 노드가 생성한 추가 질문 목록

    # ── 환자 정보 누적 ────────────────────────────────────
    patient_info: dict                    # 대화를 통해 수집된 환자 정보

    # ── 분류 근거 로그 (stage별 누적) ────────────────────
    # 각 항목 구조:
    # {
    #   "stage": 2 | 3 | 4,
    #   "selection": str,
    #   "confidence": "높음" | "중간" | "낮음",
    #   "evidence_spans": [{"quote": str, "interpretation": str}, ...],
    #   "reason": str,
    # }
    classification_log: List[dict]

    # ── 최종 결과 ─────────────────────────────────────────
    final_ktas_level: Optional[int]       # KTAS 레벨 1~5
