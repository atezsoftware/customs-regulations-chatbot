import React from "react";
import { render, screen, setupUser } from "@tests/setup/test-utils";
import type { Message } from "@/app/app/interfaces";
import type { LlmManager } from "@/lib/hooks";
import MessageToolbar from "@/app/app/message/messageComponents/MessageToolbar";
import { VoiceModeProvider } from "@/providers/VoiceModeProvider";

it("regenerates a fixed preset without exposing or changing the saved HIGH setting", async () => {
  const user = setupUser();
  const regenerate = jest.fn();
  const onRegenerate = jest.fn(() => regenerate);
  const llmManager: LlmManager = {
    currentLlm: {
      name: "Selected provider",
      provider: "openai",
      modelName: "selected-model",
    },
    defaultText: null,
    updateCurrentLlm: jest.fn(),
    temperature: 1,
    updateTemperature: jest.fn(),
    temperatureExplicitlySet: false,
    reasoningEffort: "high",
    updateReasoningEffort: jest.fn(),
    hasBoundSession: true,
    persistOverrides: async () => {},
    updateModelOverrideBasedOnChatSession: jest.fn(),
    imageFilesPresent: false,
    updateImageFilesPresent: jest.fn(),
    liveAgent: null,
    maxTemperature: 2,
    llmProviders: [],
    isLoadingProviders: false,
    hasAnyProvider: true,
  };
  const parentMessage: Message = {
    nodeId: 1,
    messageId: 1,
    message: "Kaynaklı soru",
    type: "user",
    files: [],
    toolCall: null,
    parentNodeId: null,
    packets: [],
  };
  render(
    <VoiceModeProvider>
      <MessageToolbar
        nodeId={2}
        messageId={2}
        includeMessageSwitcher={false}
        currentMessageInd={0}
        getPreviousMessage={() => undefined}
        getNextMessage={() => undefined}
        rawPackets={[]}
        finalAnswerRef={React.createRef<HTMLDivElement>()}
        onRegenerate={onRegenerate}
        parentMessage={parentMessage}
        llmManager={llmManager}
        fixedModelPreset
        citations={[]}
        documentMap={new Map()}
      />
    </VoiceModeProvider>,
    {
      swrConfig: {
        // GET /api/voice/status is unrelated to this fixed-preset action.
        fallback: {
          "/api/voice/status": { stt_enabled: false, tts_enabled: false },
        },
        revalidateOnMount: false,
      },
    }
  );

  await user.click(screen.getByRole("button", { name: "Regenerate" }));
  expect(onRegenerate).toHaveBeenCalledWith({ messageId: 2, parentMessage });
  expect(regenerate).toHaveBeenCalledWith(llmManager.currentLlm);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  expect(llmManager.updateReasoningEffort).not.toHaveBeenCalled();
  expect(llmManager.reasoningEffort).toBe("high");
});
