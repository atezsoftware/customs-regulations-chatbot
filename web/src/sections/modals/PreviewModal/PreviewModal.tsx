"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { MinimalOnyxDocument } from "@/lib/search/interfaces";
import { Modal } from "@opal/components";
import Text from "@/refresh-components/texts/Text";
import { Button } from "@opal/components";
import { SvgSimpleLoader } from "@opal/icons";
import { Section } from "@/layouts/general-layouts";
import FloatingFooter from "@/sections/modals/PreviewModal/FloatingFooter";
import mime from "mime";
import {
  getCodeLanguage,
  getDataLanguage,
  getLanguageByMime,
} from "@/lib/languages";
import { fetchChatFile } from "@/lib/chat/svc";
import { PreviewContext } from "@/sections/modals/PreviewModal/interfaces";
import { resolveVariant } from "@/sections/modals/PreviewModal/variants";

interface PreviewModalProps {
  presentingDocument: MinimalOnyxDocument;
  onClose: () => void;
}

export default function PreviewModal({
  presentingDocument,
  onClose,
}: PreviewModalProps) {
  const [fileContent, setFileContent] = useState("");
  const [fileUrl, setFileUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mimeType, setMimeType] = useState("application/octet-stream");
  const [zoom, setZoom] = useState(100);
  const requestSequence = useRef(0);
  const isCitationChunk = presentingDocument.citation_chunk_ind !== undefined;

  const variant = useMemo(
    () => resolveVariant(presentingDocument.semantic_identifier, mimeType),
    [presentingDocument.semantic_identifier, mimeType]
  );

  const language = useMemo(
    () =>
      getCodeLanguage(presentingDocument.semantic_identifier || "") ||
      getLanguageByMime(mimeType) ||
      getDataLanguage(presentingDocument.semantic_identifier || "") ||
      "plaintext",
    [mimeType, presentingDocument.semantic_identifier]
  );

  const lineCount = useMemo(() => {
    if (!fileContent) return 0;
    return fileContent.split("\n").length;
  }, [fileContent]);

  const fileSize = useMemo(() => {
    if (!fileContent) return "";
    const bytes = new TextEncoder().encode(fileContent).length;
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(2)} KB`;
    const mb = kb / 1024;
    return `${mb.toFixed(2)} MB`;
  }, [fileContent]);

  const fetchFile = useCallback(async () => {
    const requestId = ++requestSequence.current;
    const isCurrentRequest = () => requestSequence.current === requestId;
    setIsLoading(true);
    setLoadError(null);
    setFileContent("");
    const fileIdLocal =
      presentingDocument.document_id.split("__")[1] ||
      presentingDocument.document_id;
    const originalFileName =
      presentingDocument.semantic_identifier || "document";
    // Direct raw-file URL — usable for downloads without materializing a blob.
    const rawFileUrl = `/api/chat/file/${encodeURIComponent(fileIdLocal)}`;

    const updateFileUrl = (url: string) =>
      setFileUrl((prev) => {
        if (prev.startsWith("blob:")) window.URL.revokeObjectURL(prev);
        return url;
      });

    try {
      setFileName(originalFileName);

      if (presentingDocument.citation_chunk_ind !== undefined) {
        const params = new URLSearchParams({
          document_id: presentingDocument.document_id,
          chunk_id: String(presentingDocument.citation_chunk_ind),
        });
        const response = await fetch(`/api/document/chunk-info?${params}`);
        if (!response.ok) {
          throw new Error(
            `Chunk request failed with status ${response.status}`
          );
        }
        const payload: unknown = await response.json();
        if (
          typeof payload !== "object" ||
          payload === null ||
          !("content" in payload) ||
          typeof payload.content !== "string" ||
          !payload.content.trim()
        ) {
          throw new Error("Chunk response did not contain usable content");
        }
        if (!isCurrentRequest()) return;
        setMimeType("text/plain");
        setFileContent(payload.content);
        updateFileUrl("");
        return;
      }

      // Variants that render from backend-parsed content (spreadsheets) don't
      // need the raw binary blob — skip downloading the full workbook and let
      // the download button point at the raw URL directly.
      const preResolved = resolveVariant(
        presentingDocument.semantic_identifier,
        "application/octet-stream"
      );
      if (preResolved.needsParsedContent) {
        updateFileUrl(rawFileUrl);
        setMimeType(
          mime.getType(originalFileName) ?? "application/octet-stream"
        );
        const parsedResponse = await fetchChatFile(fileIdLocal, true);
        const parsedContent = await parsedResponse.text();
        if (isCurrentRequest()) setFileContent(parsedContent);
        return;
      }

      const response = await fetchChatFile(fileIdLocal);
      if (!isCurrentRequest()) return;

      // Re-resolve using the stored MIME from the response headers, which is
      // authoritative, BEFORE materializing the body as a blob.
      const rawContentType =
        response.headers.get("Content-Type") || "application/octet-stream";
      const resolvedMime =
        rawContentType === "application/octet-stream"
          ? (mime.getType(originalFileName) ?? rawContentType)
          : rawContentType;
      setMimeType(resolvedMime);

      const resolved = resolveVariant(
        presentingDocument.semantic_identifier,
        resolvedMime
      );
      if (resolved.needsParsedContent) {
        // Name alone didn't identify a spreadsheet, but the stored MIME did
        // (e.g. an xlsx with a renamed/missing display name). Discard the raw
        // workbook body and render from the parsed payload instead.
        await response.body?.cancel();
        updateFileUrl(rawFileUrl);
        const parsedResponse = await fetchChatFile(fileIdLocal, true);
        const parsedContent = await parsedResponse.text();
        if (isCurrentRequest()) setFileContent(parsedContent);
        return;
      }

      const blob = await response.blob();
      if (!isCurrentRequest()) return;
      updateFileUrl(window.URL.createObjectURL(blob));

      if (resolved.needsTextContent) {
        const textContent = await blob.text();
        if (isCurrentRequest()) setFileContent(textContent);
      }
    } catch (error) {
      if (!isCurrentRequest()) return;
      console.error(
        `Failed to load preview for chat file ${fileIdLocal}:`,
        error
      );
      if (isCitationChunk) {
        updateFileUrl("");
        setLoadError(
          "The cited chunk is unavailable or you no longer have access to it."
        );
      } else {
        // Keep a usable download link for the CURRENT file even when the
        // preview itself failed (a stale previous-file URL must never win).
        updateFileUrl(rawFileUrl);
        setLoadError("Failed to load document.");
      }
    } finally {
      if (isCurrentRequest()) setIsLoading(false);
    }
  }, [isCitationChunk, presentingDocument]);

  useEffect(() => {
    fetchFile();
  }, [fetchFile]);

  useEffect(() => {
    return () => {
      if (fileUrl.startsWith("blob:")) window.URL.revokeObjectURL(fileUrl);
    };
  }, [fileUrl]);

  const handleZoomIn = useCallback(
    () => setZoom((prev) => Math.min(prev + 25, 200)),
    []
  );
  const handleZoomOut = useCallback(
    () => setZoom((prev) => Math.max(prev - 25, 25)),
    []
  );

  const ctx: PreviewContext = useMemo(
    () => ({
      fileContent,
      fileUrl,
      fileName,
      language,
      lineCount,
      fileSize,
      zoom,
      onZoomIn: handleZoomIn,
      onZoomOut: handleZoomOut,
    }),
    [
      fileContent,
      fileUrl,
      fileName,
      language,
      lineCount,
      fileSize,
      zoom,
      handleZoomIn,
      handleZoomOut,
    ]
  );

  return (
    <Modal
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <Modal.Content
        width={variant.width}
        height={variant.height}
        preventAccidentalClose={false}
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <Modal.Header
          title={fileName || "Document"}
          description={variant.headerDescription(ctx)}
          onClose={onClose}
        />

        {/* Body — uses flex-1/min-h-0/overflow-hidden (not Modal.Body)
            so that child ScrollIndicatorDivs become the actual scroll
            container instead of the body stealing it via overflow-y-auto. */}
        <div className="flex flex-col flex-1 min-h-0 overflow-hidden w-full bg-background-tint-01">
          {isLoading ? (
            <Section>
              <SvgSimpleLoader className="h-8 w-8" />
            </Section>
          ) : loadError ? (
            <Section padding={1}>
              <Text text03 mainUiBody>
                {loadError}
              </Text>
              {fileUrl && (
                <a href={fileUrl} download={fileName}>
                  <Button>Download File</Button>
                </a>
              )}
            </Section>
          ) : isCitationChunk ? (
            <div className="h-full overflow-y-auto p-4">
              <Text
                as="p"
                mainContentBody
                text04
                className="whitespace-pre-wrap"
              >
                {fileContent}
              </Text>
            </div>
          ) : (
            variant.renderContent(ctx)
          )}
        </div>

        {!isCitationChunk && !isLoading && !loadError && (
          <FloatingFooter
            left={variant.renderFooterLeft(ctx)}
            right={variant.renderFooterRight(ctx)}
            codeBackground={variant.codeBackground}
          />
        )}
      </Modal.Content>
    </Modal>
  );
}
