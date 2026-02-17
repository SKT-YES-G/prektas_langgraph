"""
Pre-KTAS 애플리케이션 엔트리포인트

실행 방법:
    python -m pre_ktas.app

스트리밍 루프:
    - N초마다 새 입력을 받아 그래프에 전달
    - 그래프 출력(추가 질문 또는 분류 결과)을 콘솔에 출력
    - 키보드 인터럽트(Ctrl+C)로 종료
"""

import time
import threading
import sys
from langchain_openai import ChatOpenAI

from pre_ktas.graph.build_graph import build_graph, get_initial_state


# ── 설정 ──────────────────────────────────────────────────────────────────────
STREAM_INTERVAL_SECONDS = 10   # N초마다 재평가


# ── LLM 초기화 ────────────────────────────────────────────────────────────────
def create_llm():
    """사용할 LLM 을 생성한다. 환경변수 OPENAI_API_KEY 필요."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True,
    )


# ── 키보드 입력 스레드 ─────────────────────────────────────────────────────────
class KeyboardInputThread(threading.Thread):
    """백그라운드에서 사용자 키보드 입력을 수집한다."""

    def __init__(self):
        super().__init__(daemon=True)
        self._buffer: list[str] = []
        self._lock = threading.Lock()

    def run(self):
        print("[입력] 환자 정보를 입력하세요 (Enter로 제출):")
        while True:
            try:
                line = input()
                with self._lock:
                    self._buffer.append(line)
            except EOFError:
                break

    def flush(self) -> str:
        """버퍼에 쌓인 입력을 가져오고 초기화한다."""
        with self._lock:
            text = " ".join(self._buffer).strip()
            self._buffer.clear()
        return text


# ── 스트리밍 입력 시뮬레이터 ──────────────────────────────────────────────────
def get_stream_input() -> str:
    """
    실제 환경에서는 음성 전사(STT) 등 외부 스트림에서 가져온다.
    여기서는 예시로 빈 문자열을 반환 (keyboard_input 과 결합하여 사용).
    """
    return ""


# ── 출력 포맷터 ───────────────────────────────────────────────────────────────
def print_state_summary(state: dict):
    """현재 분류 상태를 콘솔에 출력한다."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [분류 결과]")
    print(f"  Stage 2 : {state.get('stage2_selection') or '미분류'}")
    print(f"  Stage 3 : {state.get('stage3_selection') or '미분류'}")
    print(f"  Stage 4 : {state.get('stage4_selection') or '미분류'}")

    questions = state.get("additional_questions") or []
    if questions:
        print(f"\n  [추가 질문]")
        for q in questions:
            print(f"  - {q}")

    messages = state.get("messages") or []
    for msg in messages[-2:]:     # 마지막 2개 메시지만
        content = getattr(msg, "content", "")
        if content:
            print(f"\n  [AI] {content}")

    print(sep)


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Pre-KTAS 실시간 분류 시스템 시작")
    print(f"  재평가 주기: {STREAM_INTERVAL_SECONDS}초")
    print("  종료: Ctrl+C")
    print("=" * 60)

    llm = create_llm()
    app = build_graph(llm)

    # 초기 state 설정
    state = get_initial_state()

    # 키보드 입력 스레드 시작
    kb_thread = KeyboardInputThread()
    kb_thread.start()

    cycle = 0
    try:
        while True:
            cycle += 1
            time.sleep(STREAM_INTERVAL_SECONDS)

            # 스트리밍 입력 + 키보드 입력 결합
            stream_text  = get_stream_input()
            keyboard_text = kb_thread.flush()
            combined_input = " ".join(filter(None, [stream_text, keyboard_text])).strip()

            if not combined_input:
                print(f"\n[사이클 {cycle}] 새 입력 없음 - 대기 중...")
                continue

            print(f"\n[사이클 {cycle}] 새 입력: {combined_input!r}")

            # state 에 user_input 업데이트 후 그래프 실행
            state["user_input"] = combined_input

            # stream_mode="values": 각 노드 실행 후 전체 state 스냅샷
            final_state = None
            for chunk in app.stream(state, stream_mode="values"):
                final_state = chunk

            if final_state:
                state = final_state
                print_state_summary(state)

    except KeyboardInterrupt:
        print("\n\n[종료] Pre-KTAS 시스템을 종료합니다.")
        print_state_summary(state)
        sys.exit(0)


if __name__ == "__main__":
    main()
