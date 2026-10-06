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

it("serializes Atez Search independently from Deep Research", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "stop after payload capture" }),
  });

  await expect(
    sendMessage({
      message: "Antrepo nedir?",
      parentMessageId: null,
      chatSessionId: "session-1",
      filters: null,
      atezSearch: true,
      deepResearch: false,
    }).next()
  ).rejects.toThrow("stop after payload capture");

  const request = jest.mocked(global.fetch).mock.calls[0]![1];
  const payload = JSON.parse(String(request?.body));
  expect(payload.atez_search).toBe(false);
  expect(payload.deep_research).toBe(false);
});

it("serializes Atez Search V2 independently from the original workflow", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "stop after payload capture" }),
  });

  await expect(
    sendMessage({
      message: "Antrepo nedir?",
      parentMessageId: null,
      chatSessionId: "session-1",
      filters: null,
      atezSearch: false,
      atezSearchV2: true,
    }).next()
  ).rejects.toThrow("stop after payload capture");

  const request = jest.mocked(global.fetch).mock.calls[0]![1];
  const payload = JSON.parse(String(request?.body));
  expect(payload.atez_search).toBe(false);
  expect(payload.atez_search_v2).toBe(false);
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

it("serializes ASv3 as an independent opt-in workflow", async () => {
  // POST /api/chat/send-chat-message: capture body before starting a stream.
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "stop after payload capture" }),
  });
  await expect(
    sendMessage({
      message: "Muafiyet şartları nelerdir?",
      parentMessageId: null,
      chatSessionId: "session-1",
      filters: null,
      atezSearchV3: true,
    }).next()
  ).rejects.toThrow("stop after payload capture");
  const request = jest.mocked(global.fetch).mock.calls[0]![1];
  const payload = JSON.parse(String(request?.body));
  expect(payload.atez_search_v3).toBe(true);
  expect(payload.atez_search).toBe(false);
  expect(payload.atez_search_v2).toBe(false);
  expect(payload.deep_research).toBe(false);
});

it("carries the explicit checkpoint owner when resuming ASv3", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "capture" }),
  });
  await expect(
    sendMessage({
      message: "Original question",
      parentMessageId: 88,
      chatSessionId: "session-1",
      filters: null,
      atezSearchV3: true,
      asv3ResumeMessageId: 101,
    }).next()
  ).rejects.toThrow("capture");
  const payload = JSON.parse(
    String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
  );
  expect(payload.asv3_resume_message_id).toBe(101);
  expect(payload.message).toBe("Original question");
  expect(payload.atez_search_v3).toBe(true);
});

it.each([undefined, false, true])(
  "serializes Experimental with the selected model and explicit external consent (%s)",
  async (asv3AllowExternal) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "capture" }),
    });
    await expect(
      sendMessage({
        message: "Question",
        parentMessageId: null,
        chatSessionId: "session-1",
        filters: null,
        experimentalResearch: true,
        asv3AllowExternal,
        modelProvider: "Vertex Gemini",
        modelProviderType: "vertex_ai",
        modelVersion: "gemini-3.8-pro",
      }).next()
    ).rejects.toThrow("capture");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    expect(payload.atez_search_v3).toBe(true);
    expect(payload.asv3_research_profile).toBe("experimental");
    expect(payload).not.toHaveProperty("asv3_parallel_research");
    expect(payload.deep_research).toBe(false);
    expect(payload.atez_search).toBe(false);
    expect(payload.atez_search_v2).toBe(false);
    expect(payload.asv3_allow_external).toBe(asv3AllowExternal === true);
    expect(payload.llm_override).toEqual({
      model_provider: "Vertex Gemini",
      model_provider_type: "vertex_ai",
      model_version: "gemini-3.8-pro",
    });
  }
);

it.each([
  [undefined, undefined],
  [101, false],
  [undefined, true],
  [101, true],
])(
  "preserves selected model and external consent in parallel send/resume (%s, %s)",
  async (asv3ResumeMessageId, asv3AllowExternal) => {
    // Capture POST /api/chat/send-chat-message without starting a model call.
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "capture" }),
    });
    await expect(
      sendMessage({
        message: "Question",
        parentMessageId: 88,
        chatSessionId: "session-1",
        filters: null,
        experimentalParallelResearch: true,
        asv3ResumeMessageId,
        asv3AllowExternal,
        modelProvider: "Vertex Gemini",
        modelProviderType: "vertex_ai",
        modelVersion: "gemini-3.8-flash",
      }).next()
    ).rejects.toThrow("capture");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    expect(payload.atez_search_v3).toBe(true);
    expect(payload.asv3_research_profile).toBe("experimental");
    expect(payload.asv3_parallel_research).toBe(true);
    expect(payload.deep_research).toBe(false);
    expect(payload.asv3_resume_message_id).toBe(asv3ResumeMessageId);
    expect(payload.asv3_allow_external).toBe(asv3AllowExternal === true);
    expect(payload.llm_override).toEqual({
      model_provider: "Vertex Gemini",
      model_provider_type: "vertex_ai",
      model_version: "gemini-3.8-flash",
    });
  }
);

it.each([
  [true, undefined, false],
  [true, false, false],
  [true, true, true],
  [false, true, false],
])(
  "limits external permission to an explicitly authorized ASv3 request (%s, %s)",
  async (atezSearchV3, asv3AllowExternal, expected) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "capture" }),
    });
    await expect(
      sendMessage({
        message: "Question",
        parentMessageId: null,
        chatSessionId: "session-1",
        filters: null,
        atezSearchV3,
        asv3AllowExternal,
      }).next()
    ).rejects.toThrow("capture");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    expect(payload.asv3_allow_external).toBe(expected);
  }
);

it.each([undefined, 101])(
  "preserves the selected named Vertex model in ASv3 send/resume (%s)",
  async (asv3ResumeMessageId) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "capture" }),
    });
    await expect(
      sendMessage({
        message: "Question",
        parentMessageId: 88,
        chatSessionId: "session-1",
        filters: null,
        atezSearchV3: true,
        asv3ResumeMessageId,
        modelProvider: "Vertex Gemini",
        modelProviderType: "vertex_ai",
        modelVersion: "gemini-3.8-pro",
      }).next()
    ).rejects.toThrow("capture");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    expect(payload.llm_override).toEqual({
      model_provider: "Vertex Gemini",
      model_provider_type: "vertex_ai",
      model_version: "gemini-3.8-pro",
    });
    expect(payload.asv3_resume_message_id).toBe(asv3ResumeMessageId);
  }
);
