import { sendMessage, type SendMessageParams } from "@/app/app/services/lib";
import legacyPayloads from "@/app/app/services/lib.asv3ExpCropped.legacyPayloads.json";

const originalFetch = global.fetch;
afterEach(() => {
  global.fetch = originalFetch;
});

const baseParams: SendMessageParams = {
  message: "Original question",
  parentMessageId: 88,
  chatSessionId: "session-1",
  filters: null,
  modelProvider: "OpenAI",
  modelProviderType: "openai",
  modelVersion: "gpt-6.1-sol",
  asv3AllowExternal: true,
};

async function capture(params: SendMessageParams): Promise<string> {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "capture" }),
  });
  await expect(sendMessage(params).next()).rejects.toThrow("capture");
  return String(jest.mocked(global.fetch).mock.calls[0]![1]?.body);
}

it.each(legacyPayloads)(
  "preserves the exact existing workflow request body: $mode",
  async ({ mode, body }) => {
    expect(await capture({ ...baseParams, ...mode })).toBe(body);
    expect(
      await capture({
        ...baseParams,
        ...mode,
        asv3ExpCropped: false,
        asv3Fast: false,
        asv3FastCropped: false,
      })
    ).toBe(body);
  }
);

it.each([undefined, false, true])(
  "sends exp cropped with the selected model and unchanged external consent (%s)",
  async (asv3AllowExternal) => {
    const payload = JSON.parse(
      await capture({ ...baseParams, asv3ExpCropped: true, asv3AllowExternal })
    );
    expect(payload).toMatchObject({
      atez_search_v3: true,
      asv3_research_profile: "normal",
      asv3_parallel_research: false,
      asv3_exp_cropped: true,
      asv3_allow_external: asv3AllowExternal === true,
      llm_override: {
        model_provider: "OpenAI",
        model_provider_type: "openai",
        model_version: "gpt-6.1-sol",
      },
    });
    for (const flag of [
      "asv3_exp_cheap",
      "asv3_guarded_experimental",
      "asv3_guardrails_v2",
      "asv3_guardrails_v3",
    ])
      expect(payload).not.toHaveProperty(flag);
  }
);

it.each([
  { experimentalResearch: true },
  { experimentalParallelResearch: true },
  { experimentalGuardrails: true },
  { experimentalGuardrailsV2: true },
  { experimentalGuardrailsV3: true },
  { deepResearch: true },
  { legalComposite: true },
  { legalReview: true },
  { supersearch: true },
])("rejects a mixed exp cropped request before sending: %j", async (mode) => {
  global.fetch = jest.fn();
  await expect(
    sendMessage({ ...baseParams, asv3ExpCropped: true, ...mode }).next()
  ).rejects.toThrow("birleştirilemez");
  expect(global.fetch).not.toHaveBeenCalled();
});

it("preserves the checkpoint owner without selecting a new workflow", async () => {
  const payload = JSON.parse(
    await capture({
      ...baseParams,
      atezSearchV3: true,
      asv3ResumeMessageId: 101,
    })
  );
  expect(payload.asv3_resume_message_id).toBe(101);
  expect(payload).not.toHaveProperty("asv3_exp_cropped");
});

it("rejects multi-model Cropped before sending any request", async () => {
  global.fetch = jest.fn();
  await expect(
    sendMessage({
      ...baseParams,
      asv3ExpCropped: true,
      llmOverrides: [
        { model_provider: "OpenAI", model_version: "gpt-6.1-sol" },
        { model_provider: "OpenAI", model_version: "another-model" },
      ],
    }).next()
  ).rejects.toThrow("tek modelle çalışır");
  expect(global.fetch).not.toHaveBeenCalled();
});
