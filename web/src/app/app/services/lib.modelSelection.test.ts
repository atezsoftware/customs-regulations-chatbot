import {
  persistLlmOverrideForChatSession,
  sendMessage,
  type SendMessageParams,
} from "@/app/app/services/lib";

import { ChatFileType } from "@/app/app/interfaces";

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
});

it.each([false, true])(
  "serializes Legal Composite only when selected (%s)",
  async (selected) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "captured" }),
    });
    await expect(
      sendMessage({
        message: "Antrepo nedir?",
        parentMessageId: null,
        chatSessionId: "session-1",
        filters: null,
        legalComposite: selected,
      }).next()
    ).rejects.toThrow("captured");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    if (selected) {
      expect(payload.legal_composite).toBe(true);
    } else {
      expect(payload).not.toHaveProperty("legal_composite");
    }
    expect(payload.atez_search).toBe(false);
    expect(payload.atez_search_v2).toBe(false);
    expect(payload.atez_search_v3).toBe(false);
    expect(payload.deep_research).toBe(false);
    expect(payload).not.toHaveProperty("asv3_parallel_research");
  }
);

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

it("serializes Experimental Guardrails as a separate bounded ASv3 opt-in", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "capture" }),
  });

  await expect(
    sendMessage({
      message: "Muafiyet şartları nelerdir?",
      parentMessageId: null,
      chatSessionId: "session-1",
      filters: null,
      experimentalGuardrails: true,
    }).next()
  ).rejects.toThrow("capture");

  const request = jest.mocked(global.fetch).mock.calls[0]![1];
  const payload = JSON.parse(String(request?.body));
  expect(payload.atez_search_v3).toBe(true);
  expect(payload.asv3_research_profile).toBe("normal");
  expect(payload.asv3_guarded_experimental).toBe(true);
  expect(payload).not.toHaveProperty("asv3_parallel_research");
  expect(payload.deep_research).toBe(false);
});

it("serializes Experimental Guardrails v2 without enabling the existing guardrails variant", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 400,
    json: async () => ({ detail: "capture-v2" }),
  });

  await expect(
    sendMessage({
      message: "Muafiyet şartları nelerdir?",
      parentMessageId: null,
      chatSessionId: "session-1",
      filters: null,
      experimentalGuardrailsV2: true,
    }).next()
  ).rejects.toThrow("capture-v2");

  const request = jest.mocked(global.fetch).mock.calls[0]![1];
  const payload = JSON.parse(String(request?.body));
  expect(payload.atez_search_v3).toBe(true);
  expect(payload.asv3_research_profile).toBe("normal");
  expect(payload.asv3_guardrails_v2).toBe(true);
  expect(payload).not.toHaveProperty("asv3_guarded_experimental");
  expect(payload).not.toHaveProperty("asv3_parallel_research");
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

it.each([false, true])(
  "sends Supersearch only when selected (%s)",
  async (selected) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "captured" }),
    });
    await expect(
      sendMessage({
        message: "Antrepo rejiminde teminat koşulları nelerdir?",
        chatSessionId: "session-1",
        parentMessageId: null,
        filters: null,
        supersearch: selected,
        modelProvider: "Vertex",
        modelProviderType: "vertex_ai",
        modelVersion: "gemini-flash",
      }).next()
    ).rejects.toThrow("captured");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    if (selected) {
      expect(payload.supersearch).toBe(true);
    } else {
      expect(payload).not.toHaveProperty("supersearch");
    }
    expect(payload.atez_search_v3).toBe(false);
    expect(payload.deep_research).toBe(false);
    expect(payload.asv3_allow_external).toBe(false);
    expect(payload).not.toHaveProperty("legal_composite");
    expect(payload.llm_override.model_version).toBe("gemini-flash");
  }
);

it.each([false, true])(
  "prevents stale tool and research settings from escaping Supersearch (%s)",
  async (deepResearch) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "captured" }),
    });
    await expect(
      sendMessage({
        message: "Soru",
        chatSessionId: "session-1",
        parentMessageId: null,
        filters: null,
        supersearch: true,
        legalComposite: true,
        atezSearchV3: true,
        deepResearch,
        experimentalResearch: true,
        experimentalParallelResearch: true,
        experimentalGuardrails: true,
        asv3AllowExternal: true,
        asv3ResumeMessageId: 41,
        forcedToolId: 9,
      }).next()
    ).rejects.toThrow("captured");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    expect(payload.supersearch).toBe(true);
    expect(payload.atez_search_v3).toBe(false);
    expect(payload.deep_research).toBe(false);
    expect(payload.asv3_research_profile).toBe("deep");
    expect(payload.asv3_allow_external).toBe(false);
    expect(payload.allowed_tool_ids).toEqual([]);
    expect(payload.forced_tool_id).toBeNull();
    expect(payload).not.toHaveProperty("legal_composite");
    expect(payload).not.toHaveProperty("asv3_parallel_research");
    expect(payload).not.toHaveProperty("asv3_guarded_experimental");
    expect(payload).not.toHaveProperty("asv3_resume_message_id");
  }
);

it("rejects Supersearch with multiple answer models", async () => {
  global.fetch = jest.fn();
  await expect(
    sendMessage({
      message: "Soru",
      chatSessionId: "session-1",
      parentMessageId: null,
      filters: null,
      supersearch: true,
      llmOverrides: [
        { model_provider: "provider", model_version: "model-1" },
        { model_provider: "provider", model_version: "model-2" },
      ],
    }).next()
  ).rejects.toThrow("Supersearch tek modelle çalışır.");
  expect(global.fetch).not.toHaveBeenCalled();
});

it.each([false, true])(
  "includes attached files and extra context only outside Supersearch (%s)",
  async (selected) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "captured" }),
    });
    const fileDescriptors = [
      { id: "user-file-id", type: ChatFileType.DOCUMENT },
    ];
    const filters = {
      source_type: ["file"],
      document_set: ["PC Külliyatı"],
      updated_at_range: { start: "2025-01-01", end: null },
    };
    await expect(
      sendMessage({
        message: "Soru",
        chatSessionId: "session-1",
        parentMessageId: null,
        filters,
        supersearch: selected,
        fileDescriptors,
        enabledToolIds: [1, 9],
        additionalContext: "Browser tab text",
      }).next()
    ).rejects.toThrow("captured");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    expect(payload.internal_search_filters).toEqual(filters);
    if (selected) {
      expect(payload).not.toHaveProperty("file_descriptors");
      expect(payload.additional_context).toBeNull();
      expect(payload.allowed_tool_ids).toEqual([]);
    } else {
      expect(payload.file_descriptors).toEqual(fileDescriptors);
      expect(payload.additional_context).toBe("Browser tab text");
      expect(payload.allowed_tool_ids).toEqual([1, 9]);
    }
  }
);
