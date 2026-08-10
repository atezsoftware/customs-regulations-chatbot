import type { ChatSession } from "@/app/app/interfaces";
import {
  isPendingChatSessionConfirmed,
  mergeFetchedAndPendingChatSessions,
} from "@/hooks/useChatSessions";

function session(id: string, model: string, name = "Chat"): ChatSession {
  return {
    id,
    name,
    persona_id: 0,
    time_created: "2026-08-10T00:00:00Z",
    time_updated: "2026-08-10T00:00:00Z",
    shared_status: "private" as ChatSession["shared_status"],
    project_id: null,
    current_alternate_model: model,
    current_temperature_override: null,
    current_reasoning_effort_override: null,
  };
}

describe("pending chat session model reconciliation", () => {
  it("keeps the selected model when a stale fetch returns the same session id", () => {
    const pending = session("new-chat", "OpenRouter__openrouter__gpt-5.2", "");
    const staleFetched = session("new-chat", "OpenAI__openai__gpt-4.1");

    expect(isPendingChatSessionConfirmed(pending, staleFetched)).toBe(false);
    expect(
      mergeFetchedAndPendingChatSessions([staleFetched], [pending])[0]
        ?.current_alternate_model
    ).toBe("OpenRouter__openrouter__gpt-5.2");
  });

  it("uses the fetched session after it confirms the selected model", () => {
    const pending = session("new-chat", "OpenRouter__openrouter__gpt-5.2", "");
    const confirmedFetched = session(
      "new-chat",
      "OpenRouter__openrouter__gpt-5.2",
      "Named Chat"
    );

    expect(isPendingChatSessionConfirmed(pending, confirmedFetched)).toBe(true);
    expect(
      mergeFetchedAndPendingChatSessions([confirmedFetched], [pending])[0]?.name
    ).toBe("Named Chat");
  });
});
