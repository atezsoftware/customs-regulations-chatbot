import { sendMessage, type SendMessageParams } from "@/app/app/services/lib";

const originalFetch = global.fetch;
afterEach(() => {
  global.fetch = originalFetch;
});
const base: SendMessageParams = {
  message: "Kaynaklı soru",
  parentMessageId: null,
  chatSessionId: "session",
  filters: null,
  modelProvider: "OpenAI",
  modelProviderType: "openai",
  modelVersion: "gpt-6.1-sol",
  asv3AllowExternal: false,
};
async function capture(params: SendMessageParams) {
  global.fetch = jest
    .fn()
    .mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "capture" }),
    });
  await expect(sendMessage(params).next()).rejects.toThrow("capture");
  return JSON.parse(String(jest.mocked(global.fetch).mock.calls[0]![1]?.body));
}

it.each([
  ["asv3Fast", "asv3_fast"],
  ["asv3FastCropped", "asv3_fast_cropped"],
] as const)(
  "sends an independent %s with the selected model",
  async (flag, wireFlag) => {
    const payload = await capture({ ...base, [flag]: true });
    expect(payload).toMatchObject({
      atez_search_v3: true,
      asv3_research_profile: "normal",
      asv3_parallel_research: false,
      [wireFlag]: true,
      asv3_allow_external: false,
      llm_override: {
        model_provider: "OpenAI",
        model_provider_type: "openai",
        model_version: "gpt-6.1-sol",
      },
    });
    expect(payload).not.toHaveProperty("asv3_exp_cropped");
    expect(payload).not.toHaveProperty(
      wireFlag === "asv3_fast" ? "asv3_fast_cropped" : "asv3_fast"
    );
  }
);

it.each(["asv3Fast", "asv3FastCropped"] as const)(
  "rejects mixed %s and multi-model requests before fetch",
  async (flag) => {
    for (const conflict of [
      { asv3ExpCropped: true },
      { deepResearch: true },
      { legalReview: true },
      { supersearch: true },
      { legalComposite: true },
      { experimentalResearch: true },
      { experimentalParallelResearch: true },
      { experimentalGuardrailsV3: true },
      { asv3Fast: true, asv3FastCropped: true },
    ]) {
      global.fetch = jest.fn();
      await expect(
        sendMessage({ ...base, [flag]: true, ...conflict }).next()
      ).rejects.toThrow("birleştirilemez");
      expect(global.fetch).not.toHaveBeenCalled();
    }
    global.fetch = jest.fn();
    await expect(
      sendMessage({
        ...base,
        [flag]: true,
        llmOverrides: [
          { model_provider: "OpenAI", model_version: "gpt-6.1-sol" },
          { model_provider: "OpenAI", model_version: "other" },
        ],
      }).next()
    ).rejects.toThrow("tek modelle çalışır");
    expect(global.fetch).not.toHaveBeenCalled();
  }
);
