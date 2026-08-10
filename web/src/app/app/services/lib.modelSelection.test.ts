import {
  persistLlmOverrideForChatSession,
  sendMessage,
  type SendMessageParams,
} from "@/app/app/services/lib";

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
});

it("serializes provider type with a named model override", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "stop after payload capture" }),
  });
  const params: SendMessageParams & { modelProviderType: string } = {
    message: "hello",
    parentMessageId: null,
    chatSessionId: "session-1",
    filters: null,
    modelProvider: "Shared Provider",
    modelProviderType: "vertex_ai",
    modelVersion: "shared-model",
  };

  await expect(sendMessage(params).next()).rejects.toThrow(
    "stop after payload capture"
  );

  const request = jest.mocked(global.fetch).mock.calls[0]![1];
  const payload = JSON.parse(String(request?.body));
  expect(payload.llm_override).toEqual({
    model_provider: "Shared Provider",
    model_provider_type: "vertex_ai",
    model_version: "shared-model",
  });
});

it("waits for the selected session model to be persisted", async () => {
  let releaseRequest: ((value: { ok: true }) => void) | undefined;
  global.fetch = jest.fn().mockReturnValue(
    new Promise<{ ok: true }>((resolve) => {
      releaseRequest = resolve;
    })
  );

  let completed = false;
  const persistence = persistLlmOverrideForChatSession(
    "session-1",
    "OpenRouter__openrouter__openai/gpt-5.2"
  ).then(() => {
    completed = true;
  });

  await Promise.resolve();
  expect(completed).toBe(false);
  expect(global.fetch).toHaveBeenCalledWith(
    "/api/chat/update-chat-session-model",
    expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({
        chat_session_id: "session-1",
        new_alternate_model: "OpenRouter__openrouter__openai/gpt-5.2",
      }),
    })
  );

  releaseRequest?.({ ok: true });
  await persistence;
  expect(completed).toBe(true);
});

it("fails loudly when the selected session model cannot be persisted", async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: false, status: 400 });

  await expect(
    persistLlmOverrideForChatSession(
      "session-1",
      "Missing__openrouter__missing-model"
    )
  ).rejects.toThrow("Failed to persist selected chat model: 400");
});
