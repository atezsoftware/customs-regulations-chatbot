import {
  act,
  deferred,
  render,
  screen,
  setupUser,
} from "@tests/setup/test-utils";
import SessionFeedbackModal from "@/sections/modals/SessionFeedbackModal";
import { useChatSessionStore } from "@/app/app/stores/useChatSessionStore";
import type { SessionFeedbackResponse } from "@/app/app/interfaces";

function savedFeedback(version: number): SessionFeedbackResponse {
  return {
    chat_session_id: "session-A",
    rating: 3,
    label: "Kısmi doğru",
    feedback_text: `Saved version ${version}`,
    version,
    time_created: "2026-10-11T12:00:00Z",
    time_updated: "2026-10-11T12:00:00Z",
    evaluated_through_message_id: null,
  };
}

describe("SessionFeedbackModal pending navigation", () => {
  beforeEach(() => {
    useChatSessionStore.setState({
      currentSessionId: "session-A",
      sessions: new Map(),
    });
    useChatSessionStore.getState().createSession("session-A");
    useChatSessionStore.getState().createSession("session-B");
  });

  afterEach(() => {
    jest.restoreAllMocks();
    act(() =>
      useChatSessionStore.setState({
        currentSessionId: null,
        sessions: new Map(),
      })
    );
  });

  it("a response from the previous A modal preserves a reopened A draft and newer saved version", async () => {
    const user = setupUser();
    const pending = deferred<Response>();
    const onClose = jest.fn();
    // Hold PUT /api/chat/chat-session/session-A/feedback across A → B → A navigation.
    const fetchSpy = jest
      .spyOn(global, "fetch")
      .mockReturnValueOnce(pending.promise);
    const original = render(
      <SessionFeedbackModal sessionId="session-A" onClose={onClose} />
    );
    await user.click(
      screen.getByRole("radio", { name: "3 yıldız: Kısmi doğru" })
    );
    await user.type(
      screen.getByLabelText("Açıklama (opsiyonel)"),
      "Original draft"
    );
    await user.click(screen.getByRole("button", { name: "Kaydet" }));
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    original.unmount();

    act(() => useChatSessionStore.getState().setCurrentSession("session-B"));
    const otherSession = render(
      <SessionFeedbackModal sessionId="session-B" onClose={onClose} />
    );
    otherSession.unmount();
    act(() => useChatSessionStore.getState().setCurrentSession("session-A"));
    render(<SessionFeedbackModal sessionId="session-A" onClose={onClose} />);
    await user.click(screen.getByRole("radio", { name: "1 yıldız: Yanlış" }));
    await user.type(
      screen.getByLabelText("Açıklama (opsiyonel)"),
      "Reopened draft"
    );

    // A newer save from another client has already reached this session's store.
    act(() =>
      useChatSessionStore
        .getState()
        .updateSessionFeedback("session-A", savedFeedback(2))
    );
    await act(async () => {
      pending.resolve({
        ok: true,
        json: async () => savedFeedback(1),
      } as Response);
      await pending.promise;
    });

    expect(onClose).not.toHaveBeenCalled();
    expect(
      screen.getByRole("dialog", { name: /^Sohbeti değerlendir/ })
    ).toBeVisible();
    expect(
      screen.getByRole("radio", { name: "1 yıldız: Yanlış" })
    ).toBeChecked();
    expect(screen.getByLabelText("Açıklama (opsiyonel)")).toHaveValue(
      "Reopened draft"
    );
    expect(
      useChatSessionStore.getState().sessions.get("session-A")?.sessionFeedback
        ?.version
    ).toBe(2);
    expect(
      useChatSessionStore.getState().sessions.get("session-B")?.sessionFeedback
    ).toBeNull();
  });
});
