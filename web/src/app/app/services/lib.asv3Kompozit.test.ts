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
  temperature: 0.8,
};

async function capture(
  params: SendMessageParams
): Promise<Record<string, unknown>> {
  // Capture POST /api/chat/send-chat-message without starting a provider run.
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "capture" }),
  });
  await expect(sendMessage(params).next()).rejects.toThrow("capture");
  return JSON.parse(String(jest.mocked(global.fetch).mock.calls[0]![1]?.body));
}

it.each([
  ["asv3Kompozit", "asv3_kompozit", "asv3_kompozit_cropped"],
  ["asv3KompozitCropped", "asv3_kompozit_cropped", "asv3_kompozit"],
] as const)(
  "sends %s with a server-owned composition and explicit external consent",
  async (flag, wireFlag, otherWireFlag) => {
    for (const consent of [undefined, false, true]) {
      const payload = await capture({
        ...base,
        [flag]: true,
        asv3AllowExternal: consent,
        llmOverrides: [{ model_provider: "OpenAI", model_version: "other" }],
      });
      expect(payload).toMatchObject({
        [wireFlag]: true,
        atez_search_v3: true,
        asv3_research_profile: "normal",
        asv3_allow_external: consent === true,
        llm_override: null,
        llm_overrides: null,
      });
      for (const unrelatedFlag of [
        otherWireFlag,
        "asv3_exp_cropped",
        "asv3_fast",
        "asv3_fast_cropped",
        "asv3_model_profile",
      ]) {
        expect(payload).not.toHaveProperty(unrelatedFlag);
      }
    }
  }
);

it.each(["asv3Kompozit", "asv3KompozitCropped"] as const)(
  "rejects conflicting %s workflow choices and answer comparisons before fetch",
  async (flag) => {
    const conflicts: Partial<SendMessageParams>[] = [
      { atezSearchV3: true },
      { asv3ExpCropped: true },
      { asv3Fast: true },
      { asv3FastCropped: true },
      { deepResearch: true },
      { legalReview: true },
      { supersearch: true },
      { legalComposite: true },
      { experimentalResearch: true },
      { experimentalParallelResearch: true },
      { experimentalGuardrails: true },
      { experimentalGuardrailsV2: true },
      { experimentalGuardrailsV3: true },
      { asv3Kompozit: true, asv3KompozitCropped: true },
      { asv3ResumeMessageId: 123 },
    ];
    for (const conflict of conflicts) {
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
          { model_provider: "OpenAI", model_version: "first" },
          { model_provider: "OpenAI", model_version: "second" },
        ],
      }).next()
    ).rejects.toThrow("sabit model bileşimi");
    expect(global.fetch).not.toHaveBeenCalled();
  }
);

it("leaves resumed composition and variant to the stored checkpoint", async () => {
  const payload = await capture({
    ...base,
    atezSearchV3: true,
    asv3ResumeMessageId: 123,
  });
  expect(payload.asv3_resume_message_id).toBe(123);
  expect(payload).not.toHaveProperty("asv3_kompozit");
  expect(payload).not.toHaveProperty("asv3_kompozit_cropped");
});
