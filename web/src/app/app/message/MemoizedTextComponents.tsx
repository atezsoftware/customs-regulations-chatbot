import React, { memo, JSX, useMemo, useCallback } from "react";
import { SourceIcon } from "@/components/SourceIcon";
import { WebResultIcon } from "@/components/WebResultIcon";
import { MinimalOnyxDocument, OnyxDocument } from "@/lib/search/interfaces";
import {
  SubQuestionDetail,
  CitationChunkMap,
  CitationMap,
} from "../interfaces";
import { ValidSources } from "@/lib/types";
import { ProjectFile } from "@/lib/projects/types";
import { StreamingCitation } from "@/app/app/services/streamingModels";
import { BlinkingBar } from "./BlinkingBar";
import Text from "@/refresh-components/texts/Text";
import SourceTag from "@/refresh-components/buttons/source-tag/SourceTag";
import {
  documentToSourceInfo,
  questionToSourceInfo,
  getDisplayNameForSource,
} from "@/refresh-components/buttons/source-tag/sourceTagUtils";
import { openCitationSource, openDocument } from "@/lib/search/utils";
import {
  getCitationDocument,
  getCitationReference,
} from "@/lib/search/citations";
import { ensureHrefProtocol } from "@/lib/utils";

interface DocumentCardProps {
  document: OnyxDocument;
  updatePresentingDocument: (document: MinimalOnyxDocument) => void;
  url?: string;
}
interface QuestionCardProps {
  question: SubQuestionDetail;
  openQuestion: (question: SubQuestionDetail) => void;
}

export const MemoizedAnchor = memo(
  ({
    docs,
    subQuestions,
    openQuestion,
    userFiles,
    citations,
    citationChunks,
    citationReferences,
    citationsPending = false,
    href,
    updatePresentingDocument,
    children,
  }: {
    subQuestions?: SubQuestionDetail[];
    openQuestion?: (question: SubQuestionDetail) => void;
    docs?: OnyxDocument[] | null;
    userFiles?: ProjectFile[] | null;
    citations?: CitationMap;
    citationChunks?: CitationChunkMap;
    citationReferences?: StreamingCitation[];
    citationsPending?: boolean;
    updatePresentingDocument: (doc: MinimalOnyxDocument) => void;
    href?: string;
    children: React.ReactNode;
  }): JSX.Element => {
    const value = children?.toString();
    if (value?.startsWith("[") && value?.endsWith("]")) {
      const match = value.match(/\[(D|Q)?(\d+)\]/);

      if (match) {
        const match_item = match[2];
        if (match_item !== undefined) {
          const isSubQuestion = match[1] === "Q";
          const isDocument = !isSubQuestion;

          const citation_num = parseInt(match_item, 10);
          const citationReference = citationReferences?.find(
            (citation) => citation.citation_num === citation_num
          );

          // Use citation map to find the correct document
          // Citations map format: {citation_num: document_id}
          // e.g., {1: "doc_abc", 2: "doc_xyz", 3: "doc_123"}
          let associatedDoc: OnyxDocument | null = null;
          if (isDocument && citationReference) {
            associatedDoc = getCitationDocument(
              citationReference,
              docs ?? [],
              href
            );
          } else if (isDocument) {
            const document_id = citations?.[citation_num];
            if (document_id) {
              const chunkInd = citationChunks?.[citation_num];
              const candidates =
                docs?.filter((d) => d.document_id === document_id) ?? [];
              associatedDoc =
                (chunkInd === undefined
                  ? candidates.length === 1
                    ? candidates[0]
                    : null
                  : candidates.find((d) => d.chunk_ind === chunkInd)) ?? null;
            }
          }

          const associatedSubQuestion = isSubQuestion
            ? subQuestions?.[citation_num - 1]
            : undefined;
          if (isSubQuestion && !associatedSubQuestion) return <></>;

          if (
            (!associatedDoc && !associatedSubQuestion) ||
            citationReference?.resolution_status === "unavailable"
          ) {
            return (
              <span data-citation-copy-text={value} role="status">
                <Text secondaryBody text03>
                  {[
                    value,
                    citationReference?.display_title?.trim(),
                    citationsPending
                      ? "Kaynak bekleniyor"
                      : "Kaynak çözümlenemedi",
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </Text>
              </span>
            );
          }

          let icon: React.ReactNode = null;
          if (associatedDoc?.source_type === "web") {
            icon = <WebResultIcon url={associatedDoc.link} />;
          } else {
            icon = (
              <SourceIcon
                sourceType={associatedDoc?.source_type as ValidSources}
                iconSize={18}
              />
            );
          }
          const associatedDocInfo = associatedDoc
            ? {
                ...associatedDoc,
                icon: icon as any,
                link: associatedDoc.link,
              }
            : undefined;

          return (
            <MemoizedLink
              updatePresentingDocument={updatePresentingDocument}
              href={href}
              document={associatedDocInfo}
              citation={citationReference}
              question={associatedSubQuestion}
              openQuestion={openQuestion}
            >
              {children}
            </MemoizedLink>
          );
        }
      }
    }
    return (
      <MemoizedLink
        updatePresentingDocument={updatePresentingDocument}
        href={href}
      >
        {children}
      </MemoizedLink>
    );
  }
);

export const MemoizedLink = memo(
  ({
    node,
    document,
    updatePresentingDocument,
    question,
    href,
    citation,
    openQuestion,
    ...rest
  }: Partial<DocumentCardProps & QuestionCardProps> & {
    node?: any;
    citation?: StreamingCitation;
    [key: string]: any;
  }) => {
    const value = rest.children;

    // Convert document to SourceInfo for SourceTag
    const documentSourceInfo = useMemo(() => {
      if (!document) return null;
      const info = documentToSourceInfo(document as OnyxDocument);
      return {
        ...info,
        title: citation?.display_title?.trim() || info.title,
        description: citation?.text ?? info.description,
        reference: citation
          ? getCitationReference(citation, document as OnyxDocument)
          : undefined,
      };
    }, [citation, document]);

    // Convert question to SourceInfo for SourceTag
    const questionSourceInfo = useMemo(() => {
      if (!question) return null;
      return questionToSourceInfo(question, question.level_question_num);
    }, [question]);

    // Handle click on SourceTag
    const handleSourceClick = useCallback(() => {
      if (document && citation && updatePresentingDocument) {
        const citedDocument = document as OnyxDocument;
        openCitationSource(
          citation,
          citedDocument,
          getDisplayNameForSource(citedDocument),
          updatePresentingDocument
        );
      } else if (document && updatePresentingDocument) {
        openDocument(document as OnyxDocument, updatePresentingDocument);
      } else if (question && openQuestion) {
        openQuestion(question);
      }
    }, [citation, document, updatePresentingDocument, question, openQuestion]);

    if (value?.toString().startsWith("*")) {
      return <BlinkingBar addMargin />;
    } else if (value?.toString().startsWith("[")) {
      const sourceInfo = documentSourceInfo || questionSourceInfo;
      if (!sourceInfo) {
        return <>{rest.children}</>;
      }

      const displayName = document
        ? citation?.display_title?.trim() ||
          getDisplayNameForSource(document as OnyxDocument)
        : question?.question || "Question";

      return (
        <span
          data-citation-copy-text={value?.toString()}
          data-citation-copy-href={
            citation?.preview_url ||
            document?.citation_preview_url ||
            href ||
            document?.link ||
            ""
          }
        >
          <SourceTag
            variant="inlineCitation"
            displayName={displayName}
            sources={[sourceInfo]}
            onSourceClick={handleSourceClick}
            showDetailsCard
            className="mr-0.5"
          />
          {documentSourceInfo?.reference && (
            <Text
              secondaryBody
              text03
              className="inline-block max-w-40 truncate align-middle"
            >
              {documentSourceInfo.reference}
            </Text>
          )}
        </span>
      );
    }

    const url = ensureHrefProtocol(href);

    // Check if the link is to a file on the backend
    const isChatFile = url?.includes("/api/chat/file/");
    if (isChatFile && updatePresentingDocument) {
      const fileId = url!.split("/api/chat/file/")[1]?.split(/[?#]/)[0] || "";
      const filename = value?.toString() || "download";
      return (
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            updatePresentingDocument({
              document_id: fileId,
              semantic_identifier: filename,
            });
          }}
          className="cursor-pointer text-link hover:text-link-hover"
        >
          {rest.children}
        </a>
      );
    }

    return (
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="cursor-pointer text-link hover:text-link-hover"
      >
        {rest.children}
      </a>
    );
  }
);

interface MemoizedParagraphProps {
  className?: string;
  children?: React.ReactNode;
}

export const MemoizedParagraph = memo(function MemoizedParagraph({
  className,
  children,
}: MemoizedParagraphProps) {
  return (
    <Text as="p" mainContentBody text04 className={className}>
      {children}
    </Text>
  );
});

MemoizedAnchor.displayName = "MemoizedAnchor";
MemoizedLink.displayName = "MemoizedLink";
MemoizedParagraph.displayName = "MemoizedParagraph";
