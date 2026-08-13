"use client";

import { useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button, Text } from "@opal/components";
import {
  SvgArrowUpRight,
  SvgBook,
  SvgChevronDown,
  SvgChevronUp,
  SvgFileText,
} from "@opal/icons";
import { cn } from "@opal/utils";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/refresh-components/Collapsible";
import { FeedbackBadge } from "../FeedbackBadge";
import type { MessageSnapshot } from "../../usage/types";
import "@/app/app/message/custom-code-styles.css";

interface QueryHistoryTranscriptProps {
  messages: MessageSnapshot[];
}

interface TranscriptMessageProps {
  message: MessageSnapshot;
}

type SourceDocument = MessageSnapshot["documents"][number];

interface SourceGroup {
  name: string;
  sources: SourceDocument[];
}

const SOURCES_PREVIEW_LIMIT = 6;

function getSourceGroupName(semanticIdentifier: string) {
  return semanticIdentifier.split(" — ")[0] || semanticIdentifier;
}

function getSourceReference(semanticIdentifier: string) {
  const [, ...referenceParts] = semanticIdentifier.split(" — ");
  return referenceParts.join(" — ");
}

function groupSourceDocuments(documents: SourceDocument[]): SourceGroup[] {
  const uniqueDocuments = documents.filter(
    (document, index, allDocuments) =>
      allDocuments.findIndex(
        (candidate) =>
          candidate.document_id === document.document_id &&
          candidate.semantic_identifier === document.semantic_identifier &&
          candidate.link === document.link
      ) === index
  );
  const groups = new Map<string, SourceDocument[]>();

  uniqueDocuments.forEach((document) => {
    const groupName = getSourceGroupName(document.semantic_identifier);
    groups.set(groupName, [...(groups.get(groupName) ?? []), document]);
  });

  return Array.from(groups, ([name, sources]) => ({ name, sources }));
}

function SourceGroup({ group }: { group: SourceGroup }) {
  const [isOpen, setIsOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const visibleSources = showAll
    ? group.sources
    : group.sources.slice(0, SOURCES_PREVIEW_LIMIT);
  const remainingSourceCount = group.sources.length - visibleSources.length;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="rounded-8 border border-border-02 bg-background-neutral-01">
        <div className="flex items-center gap-2 px-3 py-2">
          <SvgFileText className="h-4 w-4 shrink-0 text-text-03" />
          <div className="min-w-0 flex-1">
            <div className="truncate">
              <Text font="secondary-action" color="text-04">
                {group.name}
              </Text>
            </div>
            <Text font="secondary-body" color="text-03">
              {`${group.sources.length} kaynak`}
            </Text>
          </div>
          <CollapsibleTrigger asChild>
            <Button
              aria-label={`${group.name} kaynaklarını ${isOpen ? "gizle" : "göster"}`}
              icon={isOpen ? SvgChevronUp : SvgChevronDown}
              prominence="tertiary"
              size="sm"
              tooltip={isOpen ? "Kaynakları gizle" : "Kaynakları göster"}
            />
          </CollapsibleTrigger>
        </div>
        <CollapsibleContent>
          <div className="flex flex-col gap-1 border-t border-border-02 p-2">
            {visibleSources.map((source) => {
              const reference =
                getSourceReference(source.semantic_identifier) ||
                source.semantic_identifier;
              const content = (
                <>
                  <div className="line-clamp-2">
                    <Text font="secondary-body" color="text-04">
                      {reference}
                    </Text>
                  </div>
                  {source.link && (
                    <SvgArrowUpRight className="h-3.5 w-3.5 shrink-0 text-text-03" />
                  )}
                </>
              );

              return source.link ? (
                <a
                  key={`${source.document_id}-${source.semantic_identifier}-${source.link}`}
                  href={source.link}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-start gap-2 rounded-6 px-2 py-1.5 hover:bg-background-tint-02"
                >
                  <div className="min-w-0 flex-1">{content}</div>
                </a>
              ) : (
                <div
                  key={`${source.document_id}-${source.semantic_identifier}-${source.link}`}
                  className="flex items-start gap-2 rounded-6 px-2 py-1.5"
                >
                  <div className="min-w-0 flex-1">{content}</div>
                </div>
              );
            })}
            {remainingSourceCount > 0 && (
              <Button
                prominence="tertiary"
                size="sm"
                onClick={() => setShowAll(true)}
              >
                {`${remainingSourceCount} kaynak daha göster`}
              </Button>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
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
      className="prose prose-onyx max-w-none wrap-break-word font-main-content-body"
      remarkPlugins={[remarkGfm]}
      components={components}
    >
      {content}
    </ReactMarkdown>
  );
}

function ReferenceDocuments({ message }: TranscriptMessageProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (message.documents.length === 0) {
    return null;
  }

  const sourceGroups = groupSourceDocuments(message.documents);
  const sourceCount = sourceGroups.reduce(
    (count, group) => count + group.sources.length,
    0
  );

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="rounded-12 border border-border-02 bg-background-neutral-02 p-1.5">
        <CollapsibleTrigger asChild>
          <Button
            aria-label={`Kaynakları ${isOpen ? "gizle" : "göster"}`}
            icon={SvgBook}
            rightIcon={isOpen ? SvgChevronUp : SvgChevronDown}
            prominence="tertiary"
            size="sm"
            width="full"
          >
            {`Kaynaklar · ${sourceCount} sonuç · ${sourceGroups.length} belge`}
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="flex flex-col gap-2 px-1.5 pb-1.5 pt-2">
            {sourceGroups.map((group) => (
              <SourceGroup key={group.name} group={group} />
            ))}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function AssistantMessage({ message }: TranscriptMessageProps) {
  return (
    <article
      className="flex max-w-200 flex-col gap-3 self-start"
      data-testid="query-history-assistant-message"
    >
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
      <div className="whitespace-break-spaces break-anywhere">
        <Text font="main-content-body" color="text-04">
          {message.message}
        </Text>
      </div>
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
