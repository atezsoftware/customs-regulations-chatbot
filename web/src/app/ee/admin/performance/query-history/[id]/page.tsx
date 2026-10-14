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
import type { ChatSessionSnapshot } from "@/app/ee/admin/performance/usage/types";
import QueryHistoryTranscript from "@/app/ee/admin/performance/query-history/[id]/QueryHistoryTranscript";
import { SessionFeedbackBadge } from "@/app/ee/admin/performance/query-history/SessionFeedbackBadge";
import QueryHistoryUser from "@/app/ee/admin/performance/query-history/QueryHistoryUser";

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

  const modelDisplayNames = Array.from(
    new Set(
      chatSessionSnapshot.messages.flatMap((message) =>
        message.model_display_name ? [message.model_display_name] : []
      )
    )
  );

  return (
    <>
      <div className="flex flex-col gap-1 rounded-12 border border-border-02 bg-background-neutral-02 p-4">
        <Text font="main-ui-action" color="text-05">
          {chatSessionSnapshot.name || "Untitled conversation"}
        </Text>
        <QueryHistoryUser
          session={chatSessionSnapshot}
          anonymousLabel="Anonymous user"
        />
        <Text font="secondary-body" color="text-03">
          {`${chatSessionSnapshot.assistant_name || "Unknown assistant"} · ${timestampToReadableDate(chatSessionSnapshot.time_created)}`}
        </Text>
        {modelDisplayNames.length > 0 && (
          <Text font="secondary-body" color="text-03">
            {`AI modeli: ${modelDisplayNames.join(", ")}`}
          </Text>
        )}
        {chatSessionSnapshot.session_feedback && (
          <div className="flex flex-col gap-2 border-t border-border-02 pt-3">
            <Text font="main-ui-action">Sohbet değerlendirmesi</Text>
            <SessionFeedbackBadge
              feedback={chatSessionSnapshot.session_feedback}
            />
            <Text font="secondary-body" color="text-03">
              {`Son değerlendirme: ${timestampToReadableDate(chatSessionSnapshot.session_feedback.time_updated)}`}
            </Text>
            {chatSessionSnapshot.session_feedback.feedback_text && (
              <div className="whitespace-pre-wrap break-words">
                <Text font="main-ui-body" as="p">
                  {chatSessionSnapshot.session_feedback.feedback_text}
                </Text>
              </div>
            )}
          </div>
        )}
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
