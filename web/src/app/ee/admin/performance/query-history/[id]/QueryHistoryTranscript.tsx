"use client";

import { useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Text } from "@opal/components";
import { SvgArrowUpRight, SvgBook } from "@opal/icons";
import { cn } from "@opal/utils";
import { FeedbackBadge } from "../FeedbackBadge";
import type { MessageSnapshot } from "../../usage/types";

interface QueryHistoryTranscriptProps {
  messages: MessageSnapshot[];
}

interface TranscriptMessageProps {
  message: MessageSnapshot;
}

function AssistantMessageMarkdown({ content }: { content: string }) {
  const components = useMemo<Components>(
    () => ({
      a: ({ href, children }) => (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="text-action-selection-05 underline underline-offset-2"
        >
          {children}
        </a>
      ),
      code: ({ className, children }) => (
        <code
          className={cn(
            "rounded-4 bg-background-neutral-03 px-1 py-0.5 font-mono text-sm",
            className
          )}
        >
          {children}
        </code>
      ),
      pre: ({ children }) => (
        <pre className="overflow-x-auto rounded-8 bg-background-neutral-03 p-3">
          {children}
        </pre>
      ),
    }),
    []
  );

  return (
    <ReactMarkdown
      className="prose max-w-none wrap-break-word text-text-01"
      remarkPlugins={[remarkGfm]}
      components={components}
    >
      {content}
    </ReactMarkdown>
  );
}

function ReferenceDocuments({ message }: TranscriptMessageProps) {
  if (message.documents.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-2 rounded-12 border border-border-02 bg-background-neutral-02 p-3">
      <div className="flex items-center gap-1.5">
        <SvgBook className="h-4 w-4 text-text-03" />
        <Text font="secondary-action" color="text-03">
          Sources
        </Text>
      </div>
      <div className="flex flex-col gap-1">
        {message.documents.map((document) =>
          document.link ? (
            <a
              key={document.document_id}
              href={document.link}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-action-selection-05 hover:underline"
            >
              <Text font="secondary-body" color="inherit" nowrap>
                {document.semantic_identifier}
              </Text>
              <SvgArrowUpRight className="h-3.5 w-3.5 shrink-0" />
            </a>
          ) : (
            <Text
              key={document.document_id}
              font="secondary-body"
              color="text-03"
            >
              {document.semantic_identifier}
            </Text>
          )
        )}
      </div>
    </div>
  );
}

function AssistantMessage({ message }: TranscriptMessageProps) {
  return (
    <article className="flex max-w-200 flex-col gap-3 self-start" data-testid="query-history-assistant-message">
      <AssistantMessageMarkdown content={message.message} />
      <ReferenceDocuments message={message} />
      {message.feedback_type && (
        <div className="flex flex-col items-start gap-1.5">
          <FeedbackBadge feedback={message.feedback_type} />
          {message.feedback_text && (
            <Text font="secondary-body" color="text-03">
              {message.feedback_text}
            </Text>
          )}
        </div>
      )}
    </article>
  );
}

function UserMessage({ message }: TranscriptMessageProps) {
  return (
    <article
      className="max-w-150 self-end rounded-t-16 rounded-bl-16 bg-background-tint-02 px-3 py-2"
      data-testid="query-history-user-message"
    >
      <Text font="main-ui-body" color="text-01" className="whitespace-break-spaces break-anywhere">
        {message.message}
      </Text>
    </article>
  );
}

function TranscriptMessage({ message }: TranscriptMessageProps) {
  return message.message_type === "user" ? (
    <UserMessage message={message} />
  ) : (
    <AssistantMessage message={message} />
  );
}

export default function QueryHistoryTranscript({
  messages,
}: QueryHistoryTranscriptProps) {
  return (
    <section
      aria-label="Conversation history"
      className="flex w-full flex-col gap-10 py-4"
    >
      {messages.map((message) => (
        <TranscriptMessage key={message.id} message={message} />
      ))}
    </section>
  );
}
