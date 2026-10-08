import {
  act,
  cleanup,
  render,
  renderHook,
  screen,
  setupUser,
} from "@tests/setup/test-utils";
import { ReadonlyURLSearchParams } from "next/navigation";
import { ErrorBanner } from "@/app/app/message/Resubmit";
import { Message } from "@/app/app/interfaces";
import { useChatSessionStore } from "@/app/app/stores/useChatSessionStore";
import useChatController from "@/hooks/useChatController";
import { MinimalAgent } from "@/lib/agents/types";
import { FilterManager, LlmManager } from "@/lib/hooks";

const SESSION = "owned-supersearch-session";
const RUN_ID = 4503;
const MODEL = "gemini-3.8-flash";
const mockParams = new URLSearchParams("title=Saved+chat");
const mockRouter = { push: jest.fn(), replace: jest.fn() };
const mockProjects = {
  fetchProjects: jest.fn(),
  setCurrentMessageFiles: jest.fn(),
  beginUpload: jest.fn(),
};

jest.mock("next/navigation", () => ({
  usePathname: () => "/app",
  useRouter: () => mockRouter,
  useSearchParams: () => mockParams,
}));
jest.mock("@/hooks/useChatSessions", () => ({
  __esModule: true,
  default: () => ({
    refreshChatSessions: jest.fn(),
    addPendingChatSession: jest.fn(),
  }),
}));
jest.mock("@/lib/agents/hooks", () => ({
  usePinnedAgents: () => ({
    pinnedAgents: [{ id: 0 }],
    togglePinnedAgent: jest.fn(),
  }),
  useAgentPreferences: () => ({ agentPreferences: {} }),
}));
jest.mock("@/lib/hooks/useForcedTools", () => ({
  useForcedTools: () => ({ forcedToolIds: [] }),
}));
jest.mock("@/providers/ProjectsContext", () => ({
  useProjectsContext: () => mockProjects,
}));
jest.mock("@/lib/extension/utils", () => ({
  getExtensionContext: () => ({ isExtension: false, context: null }),
}));

const agent: MinimalAgent = {
  id: 0,
  name: "Assistant",
  description: "",
  tools: [],
  starter_messages: null,
  document_sets: [],
  is_public: true,
  is_listed: true,
  display_priority: null,
  is_featured: false,
  builtin_persona: true,
  owner: null,
  owner_group: null,
  user_permission: null,
};
const llmManager: LlmManager = {
  currentLlm: {
    name: "Selected provider",
    provider: "vertex_ai",
    modelName: MODEL,
  },
  defaultText: null,
  updateCurrentLlm: jest.fn(),
  temperature: 1,
  updateTemperature: jest.fn(),
  temperatureExplicitlySet: false,
  reasoningEffort: null,
  updateReasoningEffort: jest.fn(),
  hasBoundSession: true,
  persistOverrides: async () => {},
  updateModelOverrideBasedOnChatSession: jest.fn(),
  imageFilesPresent: false,
  updateImageFilesPresent: jest.fn(),
  liveAgent: agent,
  maxTemperature: 2,
  llmProviders: [],
  isLoadingProviders: false,
  hasAnyProvider: false,
};
const filters: FilterManager = {
  timeRange: null,
  selectedSources: [],
  selectedDocumentSets: [],
  selectedTags: [],
  setTimeRange: jest.fn(),
  setSelectedSources: jest.fn(),
  setSelectedDocumentSets: jest.fn(),
  setSelectedTags: jest.fn(),
  getFilterString: () => "",
  buildFiltersFromQueryString: jest.fn(),
  clearFilters: jest.fn(),
};

const accepted = {
  type: "message_id_info",
  user_message_id: 4502,
  reserved_assistant_message_id: RUN_ID,
};
function progress(status = "running") {
  return {
    placement: { turn_index: 0 },
    obj: {
      type: "asv3_progress",
      workflow: "supersearch",
      run_id: "native-run",
      event_id: status,
      sequence: 1,
      language: "tr",
      phase: status === "completed" ? "completed" : "tools",
      status,
      title: "Supersearch",
    },
  };
}
function streamResponse(packets: unknown[], disconnect: boolean): Response {
  let readIndex = 0;
  return {
    ok: true,
    body: {
      getReader: () => ({
        read: async () => {
          if (readIndex++ === 0 && packets.length) {
            return {
              done: false,
              value: new TextEncoder().encode(
                packets.map((packet) => JSON.stringify(packet)).join("\n") +
                  "\n"
              ),
            };
          }
          if (disconnect) throw new TypeError("network error");
          return { done: true };
        },
        cancel: async () => {},
      }),
    },
  } as unknown as Response;
}

const originalFetch = global.fetch;
let fetchMock: jest.MockedFunction<typeof fetch>;
function setupController(
  packets: unknown[],
  disconnect = true,
  stopStatus = 200
) {
  fetchMock = jest.fn(async (input) => {
    const url = String(input);
    if (url === "/api/chat/send-chat-message")
      return streamResponse(packets, disconnect);
    if (url.startsWith("/api/chat/stop-chat-session/")) {
      return { ok: stopStatus === 200, status: stopStatus } as Response;
    }
    return { ok: true, json: async () => ({}) } as Response;
  });
  global.fetch = fetchMock;
  const store = useChatSessionStore.getState();
  store.createSession(SESSION, { description: "Saved chat" });
  store.setCurrentSession(SESSION);
  return renderHook(
    ({ sessionId }) =>
      useChatController({
        filterManager: filters,
        llmManager,
        liveAgent: agent,
        availableAgents: [agent],
        existingChatSessionId: sessionId,
        selectedDocuments: [],
        searchParams: mockParams as ReadonlyURLSearchParams,
        resetInputBar: jest.fn(),
        setSelectedAgentFromId: jest.fn(),
      }),
    { initialProps: { sessionId: SESSION } }
  );
}

function assistant(): Message | undefined {
  return Array.from(
    useChatSessionStore
      .getState()
      .sessions.get(SESSION)
      ?.messageTree.values() ?? []
  ).find((message) => message.type === "assistant" || message.type === "error");
}
function stopCalls() {
  return fetchMock.mock.calls.filter(([input]) =>
    String(input).includes("stop-chat-session")
  );
}

afterEach(() => {
  cleanup();
  global.fetch = originalFetch;
  useChatSessionStore.setState({ sessions: new Map(), currentSessionId: null });
});

it.each([true, false])(
  "fences an accepted Supersearch run after transport failure or premature EOF (disconnect %s)",
  async (disconnect) => {
    const { result } = setupController([accepted, progress()], disconnect);
    await act(async () => {
      await result.current.onSubmit({
        message: "PC question",
        currentMessageFiles: [],
        deepResearch: false,
        supersearch: true,
      });
    });
    expect(stopCalls()).toHaveLength(1);
    expect(stopCalls()[0]?.[1]?.signal).toBeInstanceOf(AbortSignal);
    expect(assistant()).toMatchObject({
      type: "error",
      supersearch: true,
      messageId: RUN_ID,
      modelDisplayName: MODEL,
      errorCode: "CONNECTION_ERROR",
      isRetryable: false,
      errorDetails: {
        workflow: "supersearch",
        model: MODEL,
        provider: "vertex_ai",
        chat_session_id: SESSION,
        run_id: RUN_ID,
        stop_status: "requested",
      },
    });
    await act(async () => {
      await result.current.onSubmit({
        message: "Retry",
        currentMessageFiles: [],
        deepResearch: false,
        supersearch: true,
      });
    });
    expect(
      fetchMock.mock.calls.filter(([input]) =>
        String(input).includes("send-chat-message")
      )
    ).toHaveLength(1);
  }
);

it.each([{}, { legalComposite: true }, { atezSearchV3: true }])(
  "preserves other workflows on network failure (%j)",
  async (workflow) => {
    const { result } = setupController([accepted]);
    await act(async () => {
      await result.current.onSubmit({
        message: "Question",
        currentMessageFiles: [],
        deepResearch: false,
        ...workflow,
      });
    });
    expect(stopCalls()).toHaveLength(0);
    expect(assistant()?.supersearch).toBeUndefined();
    expect(assistant()?.errorCode).toBeNull();
  }
);

it("does not let a pending recovery from one chat block an unrelated chat", async () => {
  const { result, rerender } = setupController([accepted, progress()]);
  await act(async () => {
    await result.current.onSubmit({
      message: "Question",
      currentMessageFiles: [],
      deepResearch: false,
      supersearch: true,
    });
  });
  const anotherSession = "different-owned-chat";
  act(() => {
    const store = useChatSessionStore.getState();
    store.createSession(anotherSession, { description: "Another chat" });
    store.setCurrentSession(anotherSession);
    rerender({ sessionId: anotherSession });
  });
  await act(async () => {
    await result.current.onSubmit({
      message: "Different question",
      currentMessageFiles: [],
      deepResearch: false,
    });
  });
  const sends = fetchMock.mock.calls.filter(([input]) =>
    String(input).includes("send-chat-message")
  );
  expect(sends).toHaveLength(2);
  expect(JSON.parse(String(sends[1]?.[1]?.body)).chat_session_id).toBe(
    anotherSession
  );
});

it("does not fence a completed Supersearch answer when the reader fails after completion", async () => {
  const { result } = setupController([accepted, progress("completed")]);
  await act(async () => {
    await result.current.onSubmit({
      message: "Question",
      currentMessageFiles: [],
      deepResearch: false,
      supersearch: true,
    });
  });
  expect(stopCalls()).toHaveLength(0);
  expect(assistant()?.type).toBe("assistant");
});

it("preserves a structured provider error without issuing a stop request", async () => {
  const { result } = setupController(
    [
      accepted,
      {
        error: "Provider failed",
        error_code: "LLM_PROVIDER_ERROR",
        is_retryable: true,
        details: { model: MODEL, provider: "vertex_ai" },
      },
    ],
    false
  );
  await act(async () => {
    await result.current.onSubmit({
      message: "Question",
      currentMessageFiles: [],
      deepResearch: false,
      supersearch: true,
    });
  });
  expect(stopCalls()).toHaveLength(0);
  expect(assistant()).toMatchObject({
    errorCode: "LLM_PROVIDER_ERROR",
    isRetryable: true,
    errorDetails: { workflow: "supersearch", model: MODEL },
  });
});

it("does not stop a session when the send failed before an assistant run was accepted", async () => {
  const { result } = setupController([]);
  await act(async () => {
    await result.current.onSubmit({
      message: "Question",
      currentMessageFiles: [],
      deepResearch: false,
      supersearch: true,
    });
  });
  expect(stopCalls()).toHaveLength(0);
  expect(assistant()?.errorDetails?.stop_status).toBeUndefined();
});

it("does not auto-stop an explicitly aborted request", async () => {
  const { result } = setupController([]);
  fetchMock.mockImplementation(async (input) => {
    if (String(input) === "/api/chat/send-chat-message") {
      useChatSessionStore
        .getState()
        .sessions.get(SESSION)
        ?.abortController.abort();
      throw new DOMException("Aborted", "AbortError");
    }
    return { ok: true, json: async () => ({}) } as Response;
  });
  await act(async () => {
    await result.current.onSubmit({
      message: "Question",
      currentMessageFiles: [],
      deepResearch: false,
      supersearch: true,
    });
  });
  expect(stopCalls()).toHaveLength(0);
});

it("offers owner Stop retry after a failed stop and waits for terminal proof before Regenerate", async () => {
  const { result } = setupController([accepted, progress()], true, 404);
  await act(async () => {
    await result.current.onSubmit({
      message: "Question",
      currentMessageFiles: [],
      deepResearch: false,
      supersearch: true,
    });
  });
  const message = assistant()!;
  const resubmit = jest.fn();
  render(
    <ErrorBanner
      error={message.message}
      errorCode={message.errorCode || undefined}
      details={message.errorDetails || undefined}
      isRetryable={message.isRetryable}
      resubmit={resubmit}
    />
  );
  expect(screen.getByText("Supersearch bağlantı hatası")).toBeVisible();
  expect(screen.getByText(`Model: ${MODEL} (vertex_ai)`)).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Regenerate" })
  ).not.toBeInTheDocument();
  fetchMock.mockResolvedValueOnce({ ok: true } as Response);
  await setupUser().click(
    screen.getByRole("button", { name: "Çalışmayı durdur" })
  );
  expect(screen.getByText(/Durdurma isteği gönderildi/)).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Regenerate" })
  ).not.toBeInTheDocument();
  fetchMock.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      chat_session_id: SESSION,
      current_run: { run_id: RUN_ID },
      messages: [],
    }),
  } as Response);
  await setupUser().click(
    screen.getByRole("button", { name: "Durumu yenile" })
  );
  expect(
    screen.getByText("Çalışmanın bittiği henüz doğrulanmadı.")
  ).toBeVisible();
  fetchMock.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      chat_session_id: SESSION,
      current_run: null,
      messages: [
        { message_id: RUN_ID, message_type: "assistant", error: null },
      ],
      packets: [
        [
          {
            ...progress("completed"),
            obj: { ...progress("completed").obj, workflow: "asv3" },
          },
        ],
      ],
    }),
  } as Response);
  await setupUser().click(
    screen.getByRole("button", { name: "Durumu yenile" })
  );
  expect(
    screen.queryByRole("button", { name: "Regenerate" })
  ).not.toBeInTheDocument();
  fetchMock.mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      chat_session_id: SESSION,
      current_run: null,
      messages: [
        {
          message_id: RUN_ID,
          message_type: "assistant",
          error: "Research cancelled",
        },
      ],
    }),
  } as Response);
  await setupUser().click(
    screen.getByRole("button", { name: "Durumu yenile" })
  );
  await setupUser().click(screen.getByRole("button", { name: "Regenerate" }));
  expect(resubmit).toHaveBeenCalledTimes(1);
  expect(assistant()?.errorDetails?.stop_confirmed).toBe(true);
});
