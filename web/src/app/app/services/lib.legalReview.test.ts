import {
  processRawChatHistory,
  sendMessage,
  type SendMessageParams,
} from "./lib";
import { ChatFileType, type BackendMessage } from "../interfaces";
import type { Packet } from "./streamingModels";

const originalFetch = global.fetch;
const base: SendMessageParams = {
  message: "İthalatta vergilerin iadesi hangi koşullarda mümkündür?",
  parentMessageId: null,
  chatSessionId: "session-1",
  filters: null,
};

afterEach(() => {
  global.fetch = originalFetch;
});

async function capture(params: SendMessageParams) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "captured" }),
  });
  await expect(sendMessage(params).next()).rejects.toThrow("captured");
  return JSON.parse(String(jest.mocked(global.fetch).mock.calls[0]![1]?.body));
}

it("isolates Legal Review from draft attachments, tools, browser context and model overrides", async () => {
  const payload = await capture({
    ...base,
    legalReview: true,
    fileDescriptors: [{ id: "draft-file", type: ChatFileType.DOCUMENT }],
    forcedToolId: 9,
    enabledToolIds: [1, 9],
    asv3AllowExternal: true,
    additionalContext: "Browser text",
    modelProvider: "Other provider",
    modelVersion: "other-model",
    temperature: 1,
  });
  expect(payload.legal_review).toBe(true);
  expect(payload).not.toHaveProperty("file_descriptors");
  expect(payload).not.toHaveProperty("asv3_resume_message_id");
  expect(payload.allowed_tool_ids).toEqual([]);
  expect(payload.forced_tool_id).toBeNull();
  expect(payload.additional_context).toBeNull();
  expect(payload.llm_override).toBeNull();
  expect(payload.llm_overrides).toBeNull();
  expect(payload.atez_search_v3).toBe(false);
  expect(payload.deep_research).toBe(false);
  expect(payload.asv3_allow_external).toBe(false);
});

it.each<Partial<SendMessageParams>>([
  {},
  { atezSearchV3: true },
  { deepResearch: true },
  { experimentalResearch: true },
  { experimentalParallelResearch: true },
  { experimentalGuardrails: true },
  { experimentalGuardrailsV2: true },
  { experimentalGuardrailsV3: true },
  { legalComposite: true },
  { supersearch: true },
])(
  "preserves every existing workflow payload when Legal Review is off (%j)",
  async (mode) => {
    const params = {
      ...base,
      ...mode,
      modelVersion: "selected-model",
      forcedToolId: 9,
    };
    const before = await capture(params);
    const after = await capture({ ...params, legalReview: false });
    expect(after).toEqual(before);
    expect(after).not.toHaveProperty("legal_review");
  }
);

it.each<Partial<SendMessageParams>>([
  { atezSearchV3: true },
  { deepResearch: true },
  { legalComposite: true },
  { supersearch: true },
  { experimentalResearch: true },
  { experimentalParallelResearch: true },
  { experimentalGuardrails: true },
  { experimentalGuardrailsV2: true },
  { experimentalGuardrailsV3: true },
  { asv3ResumeMessageId: 41 },
])(
  "rejects mixed workflows and ASv3 checkpoints before sending (%j)",
  async (conflict) => {
    global.fetch = jest.fn();
    await expect(
      sendMessage({ ...base, legalReview: true, ...conflict }).next()
    ).rejects.toThrow("başka araştırma modlarıyla");
    expect(global.fetch).not.toHaveBeenCalled();
  }
);

it("rejects multiple answer models before sending Legal Review", async () => {
  global.fetch = jest.fn();
  await expect(
    sendMessage({
      ...base,
      legalReview: true,
      llmOverrides: [
        { model_provider: "provider", model_version: "first" },
        { model_provider: "provider", model_version: "second" },
      ],
    }).next()
  ).rejects.toThrow("tek modelle çalışır");
  expect(global.fetch).not.toHaveBeenCalled();
});

it("rehydrates Legal Review identity per historical answer without relabeling adjacent answers", () => {
  const answer: BackendMessage = {
    message_id: 2,
    message_type: "assistant",
    research_type: null,
    parent_message: 1,
    latest_child_message: null,
    message: "Kaynaklı cevap",
    rephrased_query: null,
    context_docs: [],
    time_sent: "2026-10-09T10:00:00Z",
    overridden_model: "gemini-3.8-flash",
    alternate_assistant_id: null,
    chat_session_id: "session-1",
    citations: {},
    files: [],
    tool_call: null,
    current_feedback: null,
    sub_questions: [],
    comments: null,
    parentMessageId: 1,
    refined_answer_improvement: null,
    is_agentic: null,
    preferred_response_id: null,
    model_display_name: "Gemini 3.8 Flash",
    error: null,
  };
  const packets: Packet[] = [
    {
      placement: { turn_index: 0 },
      obj: {
        type: "asv3_progress",
        workflow: "legal_review",
        run_id: "legal-run",
        event_id: "done",
        sequence: 1,
        language: "tr",
        phase: "completed",
        status: "completed",
        title: "Hukuki inceleme tamamlandı",
      },
    },
  ];
  const history = processRawChatHistory(
    [answer, { ...answer, message_id: 3 }],
    [packets, []]
  );
  expect(history.get(2)?.legalReview).toBe(true);
  expect(history.get(2)?.overridden_model).toBe("gemini-3.8-flash");
  expect(history.get(3)?.legalReview).toBeUndefined();
});
