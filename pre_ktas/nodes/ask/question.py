"""
추가 질문 노드 (Ask Question Node)

역할:
  - 재평가 노드가 '추가 질문'을 결정했을 때 호출된다.
  - state 에 쌓인 additional_questions 를 포맷하여
    AI 메시지로 대화 히스토리(messages)에 추가한다.
  - 실제 사용자 응답은 다음 N초 사이클에 user_input으로 들어온다.
"""

from langchain_core.messages import AIMessage

from pre_ktas.graph.state import GraphState


def ask_question_node(state: GraphState) -> dict:
    """추가 질문을 메시지 히스토리에 추가한다."""

    questions = state.get("additional_questions") or []

    if not questions:
        # 질문이 없으면 노드를 그냥 통과
        return {}

    # 질문들을 하나의 AI 메시지로 포맷
    formatted = _format_questions(questions)
    ai_message = AIMessage(content=formatted)

    # additional_questions 초기화 (다음 사이클에 중복 방지)
    return {
        "messages": [ai_message],
        "additional_questions": [],
    }


def _format_questions(questions: list[str]) -> str:
    """질문 목록을 사용자 친화적 형태로 포맷."""
    if len(questions) == 1:
        return f"확인이 필요합니다: {questions[0]}"

    lines = ["다음 사항을 확인해 주세요:"]
    for i, q in enumerate(questions, 1):
        lines.append(f"  {i}. {q}")
    return "\n".join(lines)
