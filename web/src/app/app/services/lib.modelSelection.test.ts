import { sendMessage, type SendMessageParams } from "@/app/app/services/lib";

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
