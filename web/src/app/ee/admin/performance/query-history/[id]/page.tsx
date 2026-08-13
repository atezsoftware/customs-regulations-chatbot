"use client";

import { use } from "react";
import { SettingsLayouts, PageLoader } from "@opal/layouts";
import { Text } from "@opal/components";
import { SvgHistory } from "@opal/icons";
import useSWR from "swr";
import { ErrorCallout } from "@/components/ErrorCallout";
import { errorHandlingFetcher } from "@/lib/fetcher";
import { SWR_KEYS } from "@/lib/swr-keys";
import { timestampToReadableDate } from "@/lib/dateUtils";
import type { ChatSessionSnapshot } from "../../usage/types";
import QueryHistoryTranscript from "./QueryHistoryTranscript";

interface QueryHistoryConversationProps {
  chatSessionId: string;
}

function QueryHistoryConversation({
  chatSessionId,
}: QueryHistoryConversationProps) {
  const {
    data: chatSessionSnapshot,
    isLoading,
    error,
  } = useSWR<ChatSessionSnapshot>(
    SWR_KEYS.adminChatSession(chatSessionId),
    errorHandlingFetcher
  );

  if (isLoading) {
    return <PageLoader />;
  }

  if (!chatSessionSnapshot || error) {
    return (
      <ErrorCallout
        errorTitle="Unable to load conversation history"
        errorMsg={`Failed to fetch chat session - ${error}`}
      />
    );
  }

  return (
    <>
      <div className="flex flex-col gap-1 rounded-12 border border-border-02 bg-background-neutral-02 p-4">
        <Text font="main-ui-action" color="text-01">
          {chatSessionSnapshot.name || "Untitled conversation"}
        </Text>
        <Text font="secondary-body" color="text-03">
          {chatSessionSnapshot.user_email || "Anonymous user"} · {" "}
          {chatSessionSnapshot.assistant_name || "Unknown assistant"} · {" "}
          {timestampToReadableDate(chatSessionSnapshot.time_created)}
        </Text>
      </div>

      <QueryHistoryTranscript messages={chatSessionSnapshot.messages} />
    </>
  );
}

export default function QueryHistoryPage(props: {
  params: Promise<{ id: string }>;
}) {
  const params = use(props.params);

  return (
    <SettingsLayouts.Root width="lg">
      <SettingsLayouts.Header
        icon={SvgHistory}
        title="Conversation History"
        description="Read-only conversation from query history"
        backButton
        divider
      />
      <SettingsLayouts.Body>
        <QueryHistoryConversation chatSessionId={params.id} />
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
