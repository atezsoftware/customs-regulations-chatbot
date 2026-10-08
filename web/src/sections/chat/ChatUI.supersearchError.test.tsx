import React from "react";
import { cleanup, render, screen, setupUser } from "@tests/setup/test-utils";
import { BackendMessage, Message } from "@/app/app/interfaces";
import { ErrorBanner } from "@/app/app/message/Resubmit";
import { processRawChatHistory } from "@/app/app/services/lib";
import { Packet } from "@/app/app/services/streamingModels";
import { useChatSessionStore } from "@/app/app/stores/useChatSessionStore";
import { ScrollContainerProvider } from "@/components/chat/ScrollContainerContext";
import { MinimalAgent } from "@/lib/agents/types";
import { LlmManager } from "@/lib/hooks";
import ChatUI from "@/sections/chat/ChatUI";

const SESSION_ID = "supersearch-failure";
const MODEL = "gemini-3.8-flash";
const FAILURE = `Error from ${MODEL}: Request timed out. Please try again.`;

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
  hasBoundSession: false,
  persistOverrides: async () => {},
  updateModelOverrideBasedOnChatSession: jest.fn(),
  imageFilesPresent: false,
  updateImageFilesPresent: jest.fn(),
  liveAgent: agent,
  maxTemperature: 2,
  llmProviders: [],
  isLoadingProviders: false,
  hasAnyProvider: true,
};

function renderChat(
  messageTree: Map<number, Message>,
  error: string | null = null
) {
  const store = useChatSessionStore.getState();
  store.createSession(SESSION_ID, { messageTree, uncaughtError: error });
  store.setCurrentSession(SESSION_ID);
  const resubmit = jest.fn();

  render(
    <ScrollContainerProvider
      scrollContainerRef={React.createRef<HTMLDivElement>()}
      contentWrapperRef={React.createRef<HTMLDivElement>()}
      spacerHeightRef={{ current: 0 }}
    >
      <ChatUI
        liveAgent={agent}
        llmManager={llmManager}
        setPresentingDocument={jest.fn()}
        onMessageSelection={jest.fn()}
        stopGenerating={jest.fn()}
        onSubmit={async () => {}}
        deepResearchEnabled={false}
        currentMessageFiles={[]}
        onResubmit={resubmit}
      />
    </ScrollContainerProvider>
  );

  return resubmit;
}

function savedFailure(message: string): BackendMessage {
  return {
    message_id: 4485,
    message_type: "assistant",
    research_type: null,
    parent_message: null,
    latest_child_message: null,
    message,
    rephrased_query: null,
    context_docs: [],
    time_sent: "2026-10-08T20:55:58Z",
    overridden_model: MODEL,
    alternate_assistant_id: null,
    chat_session_id: SESSION_ID,
    citations: null,
    files: [],
    tool_call: null,
    current_feedback: null,
    sub_questions: [],
    comments: null,
    parentMessageId: null,
    refined_answer_improvement: null,
    is_agentic: null,
    preferred_response_id: null,
    model_display_name: MODEL,
    error: FAILURE,
  };
}

function progressPacket(workflow?: "asv3" | "supersearch"): Packet {
  return {
    placement: { turn_index: 0 },
    obj: {
      type: "asv3_progress",
      workflow,
      run_id: "native-run",
      event_id: "review",
      sequence: 5,
      language: "tr",
      phase: "review",
      status: "running",
      title: "Supersearch: yanıt doğrulanıyor",
    },
  };
}

afterEach(() => {
  cleanup();
  useChatSessionStore.setState({ sessions: new Map(), currentSessionId: null });
});

it("shows a live Supersearch failure and lets the user retry without losing model provenance", async () => {
  const message: Message = {
    nodeId: -1,
    message: FAILURE,
    type: "error",
    supersearch: true,
    modelDisplayName: MODEL,
    overridden_model: MODEL,
    parentNodeId: null,
    files: [],
    toolCall: null,
    packets: [],
  };
  const resubmit = renderChat(new Map([[message.nodeId, message]]), FAILURE);

  expect(screen.getByText(FAILURE)).toBeVisible();
  await setupUser().click(screen.getByRole("button", { name: "Regenerate" }));
  expect(resubmit).toHaveBeenCalledTimes(1);
  expect(
    useChatSessionStore.getState().sessions.get(SESSION_ID)?.messageTree.get(-1)
  ).toMatchObject({ modelDisplayName: MODEL, overridden_model: MODEL });
});

it.each([FAILURE, ""])(
  "shows a saved Supersearch failure after reload with persisted message %j",
  (savedMessage) => {
    const messageTree = processRawChatHistory(
      [savedFailure(savedMessage)],
      [[progressPacket("supersearch")]]
    );
    renderChat(messageTree);

    expect(screen.getByText(FAILURE)).toBeVisible();
    expect(screen.getByRole("button", { name: "Regenerate" })).toBeEnabled();
    expect(messageTree.get(4485)).toMatchObject({
      type: "error",
      supersearch: true,
      modelDisplayName: MODEL,
      overridden_model: MODEL,
    });
  }
);

it.each(["asv3", undefined] as const)(
  "preserves the existing per-model error rendering for non-Supersearch progress %s",
  (workflow) => {
    const messageTree = processRawChatHistory(
      [savedFailure(FAILURE)],
      [[progressPacket(workflow)]]
    );
    renderChat(messageTree, FAILURE);

    expect(messageTree.get(4485)?.supersearch).toBeUndefined();
    expect(screen.queryByText(FAILURE)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Regenerate" })
    ).not.toBeInTheDocument();
  }
);

it("keeps a second failed run fenced after confirming the previous run finished", async () => {
  const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue({
    ok: true,
    json: async () => ({
      chat_session_id: SESSION_ID,
      current_run: null,
      messages: [
        { message_type: "assistant", message_id: 4485, error: "interrupted" },
      ],
    }),
  } as Response);
  const props = {
    error: "Supersearch bağlantısı kesildi.",
    errorCode: "CONNECTION_ERROR",
    resubmit: jest.fn(),
    details: {
      workflow: "supersearch",
      chat_session_id: SESSION_ID,
      run_id: 4485,
      stop_status: "requested",
    },
  };
  try {
    const { rerender } = render(<ErrorBanner {...props} />);
    await setupUser().click(
      screen.getByRole("button", { name: "Durumu yenile" })
    );
    expect(
      await screen.findByRole("button", { name: "Regenerate" })
    ).toBeEnabled();

    rerender(
      <ErrorBanner
        {...props}
        details={{
          ...props.details,
          chat_session_id: "another-supersearch-failure",
          run_id: 4503,
          stop_status: "failed",
        }}
      />
    );
    expect(
      screen.queryByRole("button", { name: "Regenerate" })
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Çalışmayı durdur" })
    ).toBeEnabled();
    expect(screen.getByRole("button", { name: "Durumu yenile" })).toBeEnabled();
    expect(props.resubmit).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  } finally {
    fetchMock.mockRestore();
  }
});
