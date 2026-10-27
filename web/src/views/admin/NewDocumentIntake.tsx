"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { Route } from "next";
import useSWR from "swr";
import { Button, InputTypeIn, Tag, Text } from "@opal/components";
import { toast } from "@opal/layouts";
import { ChunkContent } from "@/sections/cards/ChunkChangeCard";
import IntakeChunkReviewCard from "@/sections/cards/IntakeChunkReviewCard";
import IntakePredecessorReview from "@/sections/cards/IntakePredecessorReview";
import IntakeSourceIssues from "@/views/admin/IntakeSourceIssues";
import {
  type AmendmentSourcePackage,
  createAmendmentSourcePackage,
  getAmendmentSourcePackage,
  retryAmendmentSourcePackage,
  uploadAmendmentSourcePackage,
} from "@/lib/regulatory/amendments";
import {
  type DocumentIntakeBatch,
  type IntakeDocument,
  type IntakeChunkUpdate,
  type IntakeLineageSelection,
  approveDocumentIntakeBatch,
  createDocumentIntakeBatch,
  getDocumentIntakeBatch,
  getDocumentIntakeCapabilities,
  getDocumentIntakeChunks,
  getDocumentIntakeLabelingSetup,
  getDocumentIntakeRepreparePreview,
  listDocumentIntakeBatches,
  reprepareDocumentIntakeBatch,
  retryDocumentIntakeBatch,
  uploadDocumentIntakeMarkdownPackage,
  uploadDocumentIntakeZipPackage,
  updateDocumentIntakeChunk,
  updateDocumentIntakeChunks,
  saveDocumentIntakePredecessorSelection,
} from "@/lib/regulatory/documentIntake";

import {
  type IntakeChunkReviewDraft,
  type IntakePendingChunkEdit,
  hasIntakeLabelingProblem,
  intakeChunkDraftUpdate,
  intakeChunkDraftValidation,
} from "@/lib/regulatory/intakeChunkDraft";

const sourceModes = [
  { value: "file", label: "File" },
  { value: "url", label: "URL" },
  { value: "markdown", label: "Markdown package" },
] as const;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}

function labelText(value: unknown): string {
  if (Array.isArray(value))
    return value.map(labelText).filter(Boolean).join(", ");
  if (value && typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => `${key}: ${labelText(item)}`)
      .join(" · ");
  }
  return value == null ? "" : String(value);
}

function Labels({
  value,
  pending = false,
  failed = false,
}: {
  value: unknown;
  pending?: boolean;
  failed?: boolean;
}) {
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === 1 &&
    "labels" in value
  ) {
    return <Labels value={value.labels} pending={pending} failed={failed} />;
  }
  if (!value || (typeof value === "object" && !Object.keys(value).length)) {
    return (
      <Text font="secondary-body" color="text-03">
        {failed
          ? "Labeling failed · see the chunk error above."
          : pending
            ? "Labels pending."
            : "No labels assigned."}
      </Text>
    );
  }
  const entries: (readonly [string, unknown])[] = Array.isArray(value)
    ? value.map((label, index) => [String(index + 1), label] as const)
    : typeof value === "object"
      ? Object.entries(value)
      : [["Labels", value]];
  return (
    <div className="flex flex-wrap gap-2">
      {entries.map(([key, label]) => (
        <Tag
          key={key}
          title={
            Array.isArray(value)
              ? labelText(label)
              : `${key}: ${labelText(label)}`
          }
        />
      ))}
    </div>
  );
}

interface DocumentPreviewProps {
  batch: DocumentIntakeBatch;
  document: IntakeDocument;
  navigation: { documentId: string; version: number } | null;
  onNavigate: (documentId: string) => void;
  busy: boolean;
  dirtyKeys: Set<string>;
  onDirtyChange: (key: string, dirty: boolean) => void;
  chunkDrafts: Record<string, IntakePendingChunkEdit>;
  draftEpoch: number;
  onChunkDraftChange: (
    documentId: string,
    chunkId: string,
    draft: IntakeChunkReviewDraft | null
  ) => void;
  onSaveChunk: (
    documentId: string,
    chunkId: string,
    draft: IntakeChunkUpdate,
    expectedReviewSha256: string
  ) => Promise<void>;
  onSaveLineage: (
    documentId: string,
    selection: IntakeLineageSelection,
    expectedReviewSha256: string
  ) => Promise<void>;
}

function documentCardId(documentId: string): string {
  return `intake-document-${documentId}`;
}

function normalizedDocumentHref(href: string): string {
  const unescaped = href.replace(/\\([^\w\s])/g, "$1");
  try {
    return decodeURIComponent(unescaped);
  } catch {
    return unescaped;
  }
}

function DocumentPreview({
  batch,
  document,
  navigation,
  onNavigate,
  busy,
  dirtyKeys,
  onDirtyChange,
  onSaveChunk,
  onSaveLineage,
  chunkDrafts,
  draftEpoch,
  onChunkDraftChange,
}: DocumentPreviewProps) {
  const [expanded, setExpanded] = useState(true);
  const [offset, setOffset] = useState(0);
  const editReceipt =
    batch.edit_review_sha256 ??
    (batch.status === "ready" ? batch.review_sha256 : null);
  const onDraftChange = useCallback(
    (chunkId: string, draft: IntakeChunkReviewDraft | null) =>
      onChunkDraftChange(document.id, chunkId, draft),
    [document.id, onChunkDraftChange]
  );
  const cardRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (navigation?.documentId !== document.id) return;
    setExpanded(true);
    setOffset(0);
    cardRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    cardRef.current?.focus({ preventScroll: true });
  }, [navigation, document.id]);
  const { data, error } = useSWR(
    expanded && document.chunk_count > 0
      ? [
          "intake-chunks",
          batch.id,
          document.id,
          editReceipt,
          batch.review_sha256,
          draftEpoch,
          offset,
        ]
      : null,
    () => getDocumentIntakeChunks(batch.id, document.id, offset),
    {
      keepPreviousData: true,
      refreshInterval: batch.status === "preparing" ? 1500 : 0,
    }
  );
  const parent = batch.documents.find(
    (item) => item.id === document.parent_document_id
  );
  const documentLinks = (document.document_links ?? []).filter((link) =>
    batch.documents.some((item) => item.id === link.target_document_id)
  );
  const relatedDocuments = Array.from(
    new Map(
      documentLinks
        .filter((link) => link.target_document_id !== document.id)
        .map((link) => [link.target_document_id, link])
    ).values()
  );
  return (
    <section
      ref={cardRef}
      id={documentCardId(document.id)}
      tabIndex={-1}
      aria-label={document.display_name}
      className="flex flex-col gap-3 rounded-12 border border-border-02 p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Text as="h3" font="main-ui-action">
          {document.display_name}
        </Text>
        <div className="flex flex-wrap gap-2">
          <Tag title="New legislation" />
          <Tag title={document.status.replaceAll("_", " ")} />
        </div>
      </div>
      <Text font="secondary-body" color="text-03">
        {document.role === "root"
          ? "Main document"
          : `Attachment to ${parent?.display_name ?? "main document"}`}
      </Text>
      {(parent || relatedDocuments.length > 0) && (
        <div className="flex flex-wrap gap-2">
          {parent && (
            <Button
              prominence="tertiary"
              href={`#${documentCardId(parent.id)}`}
              onClick={() => onNavigate(parent.id)}
              disabled={busy}
            >
              {`Back to ${parent.display_name}`}
            </Button>
          )}
          {relatedDocuments.map((link) => (
            <Button
              key={link.target_document_id}
              prominence="tertiary"
              href={`#${documentCardId(link.target_document_id)}`}
              onClick={() => onNavigate(link.target_document_id)}
              disabled={busy}
            >
              {`Open ${link.label || "related document"}`}
            </Button>
          ))}
        </div>
      )}
      {document.source_url && (
        <Text
          font="secondary-body"
          color="text-03"
        >{`Source: ${document.source_url}`}</Text>
      )}
      <Text font="main-ui-action">{`${document.chunk_count} added chunks`}</Text>
      {Boolean(document.labeling_failed_count) && (
        <Text as="p" color="status-error-05">
          {`${document.labeling_failed_count} chunks with failed labeling in this file`}
        </Text>
      )}
      <Text font="secondary-action">File labels</Text>
      <Labels value={document.labels} pending={batch.status === "preparing"} />
      {document.error && (
        <Text as="p" color="status-error-05">
          {document.error}
        </Text>
      )}
      {document.chunk_count > 0 && (
        <Button
          prominence="tertiary"
          disabled={busy}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Hide chunks" : "Show added chunks"}
        </Button>
      )}
      {expanded && document.chunk_count > 0 && (
        <>
          {error && <Text color="status-error-05">{errorMessage(error)}</Text>}
          {!data && !error && <Text>Loading chunks…</Text>}
          {data?.items.map((chunk) => (
            <IntakeChunkReviewCard
              key={`${chunk.id}:${draftEpoch}`}
              chunk={chunk}
              documentName={document.display_name}
              sourceUrl={document.source_url}
              reviewSha256={editReceipt}
              editable={
                ["ready", "partial"].includes(batch.status) &&
                Boolean(editReceipt)
              }
              disabled={
                busy ||
                Array.from(dirtyKeys).some((key) => key.startsWith("lineage:"))
              }
              persistedDraft={chunkDrafts[chunk.id]?.draft}
              onDraftChange={onDraftChange}
              saveDisabled={Object.keys(chunkDrafts).length > 1}
              onSave={(draft, expectedReviewSha256) =>
                onSaveChunk(document.id, chunk.id, draft, expectedReviewSha256)
              }
            >
              <ChunkContent
                text={chunk.text}
                components={{
                  a: ({ children, href, node: _node, ...props }) => {
                    if (!href) return <>{children}</>;
                    const linkedDocument = documentLinks.find(
                      (link) =>
                        normalizedDocumentHref(link.href) ===
                        normalizedDocumentHref(href)
                    );
                    const className =
                      "text-link hover:text-link-hover underline underline-offset-2";
                    if (linkedDocument)
                      return (
                        <a
                          {...props}
                          className={className}
                          href={`#${documentCardId(linkedDocument.target_document_id)}`}
                          onClick={() =>
                            onNavigate(linkedDocument.target_document_id)
                          }
                        >
                          {children}
                        </a>
                      );
                    if (href.startsWith("/") || href.startsWith("#"))
                      return (
                        <Link
                          {...props}
                          className={className}
                          href={href as Route}
                        >
                          {children}
                        </Link>
                      );
                    return (
                      <a
                        {...props}
                        className={className}
                        href={href}
                        {...(/^https?:/i.test(href)
                          ? { target: "_blank", rel: "noopener noreferrer" }
                          : {})}
                      >
                        {children}
                      </a>
                    );
                  },
                }}
              />
              <Text font="secondary-action">Chunk labels</Text>
              <Labels
                value={chunk.chunk_metadata.labels}
                pending={batch.status === "preparing"}
                failed={hasIntakeLabelingProblem(chunk)}
              />
              <Text
                font="secondary-body"
                color="text-03"
              >{`Type: ${chunk.chunk_type ?? "text"} · Validity: ${chunk.validity_start_date ?? "—"} → ${chunk.validity_end_date ?? "open"}`}</Text>
            </IntakeChunkReviewCard>
          ))}
          {data && data.total > data.limit && (
            <div className="flex items-center gap-2">
              <Button
                prominence="secondary"
                disabled={offset === 0 || busy}
                onClick={() => setOffset(Math.max(0, offset - data.limit))}
              >
                Previous chunks
              </Button>
              <Text font="secondary-body">{`${offset + 1}–${Math.min(offset + data.items.length, data.total)} of ${data.total}`}</Text>
              <Button
                prominence="secondary"
                disabled={offset + data.limit >= data.total || busy}
                onClick={() => setOffset(offset + data.limit)}
              >
                Next chunks
              </Button>
            </div>
          )}
        </>
      )}
      {document.chunk_count === 0 && (
        <Text font="secondary-body" color="text-03">
          {batch.status === "preparing"
            ? "Chunks are being prepared…"
            : "No chunks are available for this document."}
        </Text>
      )}
      <IntakePredecessorReview
        batch={batch}
        document={document}
        disabled={
          busy ||
          (dirtyKeys.size > 0 && !dirtyKeys.has(`lineage:${document.id}`))
        }
        onDirtyChange={onDirtyChange}
        onSave={(selection, expectedReviewSha256) =>
          onSaveLineage(document.id, selection, expectedReviewSha256)
        }
      />
    </section>
  );
}

export default function NewDocumentIntake() {
  const { data: capability, error: capabilityError } = useSWR(
    "document-intake-capability",
    getDocumentIntakeCapabilities
  );
  const targetId = capability?.enabled
    ? capability.target_document_set_id
    : null;
  const { data: setup, error: setupError } = useSWR(
    targetId ? ["intake-labeling-setup", targetId] : null,
    getDocumentIntakeLabelingSetup
  );
  const {
    data: history,
    error: historyError,
    mutate: refreshHistory,
  } = useSWR(
    targetId ? "document-intake-history" : null,
    listDocumentIntakeBatches
  );
  const [mode, setMode] = useState<"file" | "url" | "markdown">("file");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [initialPackage, setInitialPackage] =
    useState<AmendmentSourcePackage | null>(null);
  const [initialBatch, setInitialBatch] = useState<DocumentIntakeBatch | null>(
    null
  );
  const [navigation, setNavigation] =
    useState<DocumentPreviewProps["navigation"]>(null);
  const [busy, setBusy] = useState(false);
  const [lineageDirtyKeys, setDirtyKeys] = useState<Set<string>>(
    () => new Set()
  );
  const [chunkDrafts, setChunkDrafts] = useState<
    Record<string, IntakePendingChunkEdit>
  >({});
  const [draftEpoch, setDraftEpoch] = useState(0);
  const pendingChunkEdits = Object.values(chunkDrafts);
  const dirtyKeys = new Set(
    Array.from(lineageDirtyKeys).concat(Object.keys(chunkDrafts))
  );
  const onChunkDraftChange = useCallback(
    (
      documentId: string,
      chunkId: string,
      draft: IntakeChunkReviewDraft | null
    ) => {
      setChunkDrafts((current) => {
        if (!draft) {
          if (!(chunkId in current)) return current;
          const next = { ...current };
          delete next[chunkId];
          return next;
        }
        if (current[chunkId]?.draft === draft) return current;
        return { ...current, [chunkId]: { documentId, chunkId, draft } };
      });
    },
    []
  );
  const onDirtyChange = useCallback((key: string, dirty: boolean) => {
    setDirtyKeys((current) => {
      if (current.has(key) === dirty) return current;
      const next = new Set(current);
      if (dirty) next.add(key);
      else next.delete(key);
      return next;
    });
  }, []);
  const [requestError, setRequestError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const requestVersion = useRef(0);
  const intakeRequestKeys = useRef(new Map<string, string>());
  const reprepareRequestKeys = useRef(new Map<string, string>());
  const {
    data: packageData,
    error: packageError,
    mutate: refreshPackage,
  } = useSWR(
    initialPackage && targetId
      ? ["intake-source", targetId, initialPackage.id]
      : null,
    () => getAmendmentSourcePackage(targetId!, initialPackage!.id),
    {
      refreshInterval: (value) =>
        (value ?? initialPackage)?.status === "processing" ? 1500 : 0,
    }
  );
  const sourcePackage =
    packageData?.id === initialPackage?.id ? packageData : initialPackage;
  const {
    data: batchData,
    error: batchError,
    mutate: refreshBatch,
  } = useSWR(
    initialBatch ? ["document-intake-batch", initialBatch.id] : null,
    () => getDocumentIntakeBatch(initialBatch!.id),
    {
      refreshInterval: (value) =>
        ["preparing", "approving"].includes(
          (value ?? initialBatch)?.status ?? ""
        )
          ? 2000
          : 0,
    }
  );
  const batch = batchData?.id === initialBatch?.id ? batchData : initialBatch;
  const reprepareAvailable = Boolean(
    batch && ["ready", "partial", "failed"].includes(batch.status)
  );
  const { data: repreparePreview, error: repreparePreviewError } = useSWR(
    reprepareAvailable && batch
      ? [
          "intake-reprepare",
          batch.id,
          batch.status,
          batch.stage,
          batch.edit_review_sha256,
          batch.review_sha256,
          draftEpoch,
        ]
      : null,
    () => getDocumentIntakeRepreparePreview(batch!.id)
  );
  useEffect(() => {
    if (
      !batch ||
      !history?.some((item) => item.id === batch.id && item !== batch)
    )
      return;
    void refreshHistory(
      (current) =>
        current?.map((item) => (item.id === batch.id ? batch : item)),
      false
    );
  }, [batch, history, refreshHistory]);
  const configurationErrors = setup?.configuration_errors ?? [];
  const labelingReady =
    Boolean(setup?.providers.length) && configurationErrors.length === 0;
  const isZipFile = file?.name.toLowerCase().endsWith(".zip") ?? false;
  useEffect(
    () => () => {
      requestVersion.current += 1;
    },
    []
  );

  function resetSource() {
    requestVersion.current += 1;
    setInitialPackage(null);
    setInitialBatch(null);
    setNavigation(null);
    setDirtyKeys(new Set());
    setChunkDrafts({});
    setDraftEpoch((current) => current + 1);
    setRequestError(null);
    setBusy(false);
  }

  async function run(action: (isCurrent: () => boolean) => Promise<void>) {
    const version = requestVersion.current;
    setBusy(true);
    setRequestError(null);
    try {
      await action(() => requestVersion.current === version);
    } catch (error) {
      if (requestVersion.current === version)
        setRequestError(errorMessage(error));
    } finally {
      if (requestVersion.current === version) setBusy(false);
    }
  }

  async function prepareSource() {
    if (!targetId) return;
    const version = requestVersion.current;
    await run(async () => {
      const key = globalThis.crypto.randomUUID();
      const prepared =
        mode === "url"
          ? await createAmendmentSourcePackage(targetId, key, {
              url: url.trim(),
            })
          : mode === "markdown"
            ? await uploadDocumentIntakeMarkdownPackage(file!, key)
            : isZipFile
              ? await uploadDocumentIntakeZipPackage(file!, key)
              : await uploadAmendmentSourcePackage(
                  targetId,
                  key,
                  file!,
                  baseUrl
                );
      if (requestVersion.current === version) setInitialPackage(prepared);
    });
  }

  async function prepareDocuments() {
    if (sourcePackage?.status !== "ready" || !labelingReady) return;
    const version = requestVersion.current;
    await run(async () => {
      const requestIdentity = sourcePackage.id;
      const key =
        intakeRequestKeys.current.get(requestIdentity) ??
        globalThis.crypto.randomUUID();
      intakeRequestKeys.current.set(requestIdentity, key);
      const created = await createDocumentIntakeBatch(sourcePackage.id, key);
      if (requestVersion.current !== version) return;
      setInitialBatch(created);
      await refreshHistory();
    });
  }

  async function saveReview(
    action: () => Promise<DocumentIntakeBatch>,
    clearChunkDrafts = false
  ) {
    const version = requestVersion.current;
    setBusy(true);
    try {
      const reviewed = await action();
      if (version !== requestVersion.current) return;
      setInitialBatch(reviewed);
      if (clearChunkDrafts) {
        setChunkDrafts({});
        setDraftEpoch((current) => current + 1);
      }
      await refreshBatch(reviewed, false);
    } finally {
      if (version === requestVersion.current) setBusy(false);
    }
  }

  const chunkSaveValidation = pendingChunkEdits
    .map((edit) => intakeChunkDraftValidation(edit.draft))
    .find(Boolean);
  const pendingReceipts = new Set(
    pendingChunkEdits.map((edit) => edit.draft.reviewSha256)
  );
  const chunkSaveError =
    pendingChunkEdits.length > 256
      ? "Save at most 256 chunk changes together."
      : pendingReceipts.size > 1 || pendingReceipts.has(null)
        ? "These drafts belong to different review versions. Discard and review the changed chunks again."
        : chunkSaveValidation;

  async function saveAllChunkChanges() {
    if (!batch || busy || chunkSaveError || !pendingChunkEdits.length) return;
    setRequestError(null);
    try {
      await saveReview(
        () =>
          updateDocumentIntakeChunks(
            batch.id,
            pendingChunkEdits[0]!.draft.reviewSha256!,
            pendingChunkEdits.map((edit) => ({
              document_id: edit.documentId,
              chunk_id: edit.chunkId,
              updates: intakeChunkDraftUpdate(edit.draft),
            }))
          ),
        true
      );
    } catch (cause) {
      setRequestError(errorMessage(cause));
    }
  }

  async function reprepareChunks() {
    if (
      !batch ||
      busy ||
      dirtyKeys.size > 0 ||
      !repreparePreview?.eligible ||
      !repreparePreview.expected_sha256
    )
      return;
    const batchId = batch.id;
    const expectedSha256 = repreparePreview.expected_sha256;
    const identity = `${batchId}:${expectedSha256}`;
    const key =
      reprepareRequestKeys.current.get(identity) ??
      globalThis.crypto.randomUUID();
    reprepareRequestKeys.current.set(identity, key);
    await run(async (isCurrent) => {
      const prepared = await reprepareDocumentIntakeBatch(
        batchId,
        expectedSha256,
        key
      );
      if (!isCurrent()) return;
      setInitialBatch(prepared);
      setDraftEpoch((current) => current + 1);
      await refreshBatch(prepared, false);
    });
  }

  if (capabilityError)
    return <Text color="status-error-05">{errorMessage(capabilityError)}</Text>;
  if (!capability) return <Text>Checking new document import…</Text>;
  if (!capability.enabled || !targetId)
    return (
      <Text color="status-error-05">
        {capability.error ?? "New document import is unavailable."}
      </Text>
    );

  return (
    <div className="flex flex-col gap-4">
      <Text as="p">{`New documents and their attachments will be added to ${capability.target_document_set_name}. Review the separate documents, chunks and labels before approving all new documents.`}</Text>
      {!batch && (
        <>
          <div className="flex gap-2">
            {sourceModes.map(({ value, label }) => (
              <Button
                key={value}
                prominence={mode === value ? "primary" : "secondary"}
                disabled={dirtyKeys.size > 0}
                onClick={() => {
                  resetSource();
                  setFile(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                  setMode(value);
                }}
              >
                {label}
              </Button>
            ))}
          </div>
          {mode === "url" ? (
            <InputTypeIn
              aria-label="New document source URL"
              placeholder="https://www.resmigazete.gov.tr/..."
              value={url}
              variant={dirtyKeys.size > 0 ? "disabled" : "primary"}
              onChange={(event) => {
                resetSource();
                setUrl(event.target.value);
              }}
            />
          ) : (
            <div className="flex flex-col gap-2">
              <input
                ref={fileInputRef}
                className="hidden"
                type="file"
                disabled={dirtyKeys.size > 0}
                accept={
                  mode === "markdown"
                    ? ".zip,.md"
                    : ".zip,.pdf,.docx,.html,.htm,.xlsx,.png,.jpg,.jpeg,.webp,.tif,.tiff"
                }
                aria-label={
                  mode === "markdown"
                    ? "Markdown package file"
                    : "New document source file"
                }
                onChange={(event) => {
                  resetSource();
                  setFile(event.target.files?.[0] ?? null);
                }}
              />
              <div className="flex items-center gap-2">
                <Button
                  prominence="secondary"
                  onClick={() => fileInputRef.current?.click()}
                >
                  {mode === "markdown"
                    ? "Choose Markdown or ZIP"
                    : "Choose document or ZIP"}
                </Button>
                {file && <Text>{file.name}</Text>}
              </div>
              {mode === "file" && (
                <Text as="p" font="secondary-body">
                  Upload a document or a ZIP containing supported documents.
                  Each ZIP member is processed according to its own format;
                  Markdown members skip conversion. Local attachment links
                  preserve relationships between separate legislation records.
                  Document names come from source titles extracted during
                  metadata preparation.
                </Text>
              )}
              {mode === "markdown" ? (
                <div className="flex flex-col gap-2">
                  <Text as="p" font="secondary-body">
                    Upload one .md document, or a ZIP with one main document and
                    its attachments. Each Markdown document becomes separate
                    legislation. Conversion is skipped; Gemini 3.5 Flash
                    generates labels before review.
                  </Text>
                  <Text as="p" font="secondary-body">
                    The ZIP must contain manifest.json and the listed .md files.
                    Each attachment needs a parent in the manifest and a
                    relative Markdown link from that parent, such as
                    [EK-1](ekler/ek-1.md). One enclosing folder is allowed.
                  </Text>
                  <div>
                    <Button
                      prominence="tertiary"
                      href="/api/regulatory/document-intake/markdown-template"
                    >
                      Download Markdown package template
                    </Button>
                  </div>
                </div>
              ) : !isZipFile ? (
                <InputTypeIn
                  aria-label="Original document URL"
                  placeholder="Original document URL (for relative attachment links)"
                  value={baseUrl}
                  variant={dirtyKeys.size > 0 ? "disabled" : "primary"}
                  onChange={(event) => {
                    resetSource();
                    setBaseUrl(event.target.value);
                  }}
                />
              ) : null}
            </div>
          )}
          <div>
            <Button
              disabled={
                busy ||
                dirtyKeys.size > 0 ||
                sourcePackage?.status === "processing" ||
                (mode === "url" ? !url.trim() : !file)
              }
              onClick={() => void prepareSource()}
            >
              Prepare source and attachments
            </Button>
          </div>
          {sourcePackage && (
            <section className="flex flex-col gap-2 rounded-12 border border-border-02 p-3">
              <Text font="main-ui-action">{`Source package ${sourcePackage.status} · ${sourcePackage.asset_count} documents`}</Text>
              <IntakeSourceIssues source={sourcePackage} />
              {sourcePackage.assets.map((asset) => (
                <Text key={asset.id} font="secondary-body">
                  {asset.display_name}
                </Text>
              ))}
              {["partial", "blocked", "failed"].includes(
                sourcePackage.status
              ) &&
                sourcePackage.issues.some(
                  (issue) => issue.retryable !== false
                ) && (
                  <div>
                    <Button
                      disabled={busy}
                      prominence="secondary"
                      onClick={() =>
                        void run(async (isCurrent) => {
                          const retried = await retryAmendmentSourcePackage(
                            targetId,
                            sourcePackage.id
                          );
                          if (!isCurrent()) return;
                          setInitialPackage(retried);
                          await refreshPackage(retried, false);
                        })
                      }
                    >
                      Retry source preparation
                    </Button>
                  </div>
                )}
            </section>
          )}
          <div className="flex flex-col gap-1">
            <Text font="secondary-action">Labeling model</Text>
            <Text font="main-ui-action">Gemini 3.5 Flash</Text>
          </div>
          {setup && !setup.providers.length && (
            <Text color="status-error-05">
              Configure a labeling connection before preparing documents.
            </Text>
          )}
          {configurationErrors.map((message) => (
            <Text key={message} color="status-error-05">
              {message}
            </Text>
          ))}
        </>
      )}
      {busy && <Text font="secondary-body">Working…</Text>}
      {[
        requestError,
        setupError && errorMessage(setupError),
        packageError && errorMessage(packageError),
        batchError && errorMessage(batchError),
        historyError && errorMessage(historyError),
      ]
        .filter(Boolean)
        .map((message, index) => (
          <Text key={index} color="status-error-05">
            {String(message)}
          </Text>
        ))}
      {!batch && (
        <div>
          <Button
            disabled={
              busy ||
              sourcePackage?.status !== "ready" ||
              !labelingReady ||
              configurationErrors.length > 0 ||
              !setup?.default_label_count ||
              Boolean(batch)
            }
            onClick={() => void prepareDocuments()}
          >
            Prepare new documents
          </Button>
        </div>
      )}
      {history && history.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {history.map((item) => (
            <Button
              key={item.id}
              size="sm"
              prominence={batch?.id === item.id ? "primary" : "secondary"}
              disabled={dirtyKeys.size > 0 || busy}
              onClick={() => {
                requestVersion.current += 1;
                setInitialBatch(item);
                setDirtyKeys(new Set());
                setChunkDrafts({});
                setDraftEpoch((current) => current + 1);
                setBusy(false);
                setRequestError(null);
              }}
            >{`${item.documents[0]?.display_name ?? "New documents"} (${item.status})`}</Button>
          ))}
        </div>
      )}
      {batch && (
        <section className="flex flex-col gap-3">
          <div>
            <Button
              prominence="tertiary"
              disabled={dirtyKeys.size > 0 || busy}
              onClick={() => {
                resetSource();
                setFile(null);
                setUrl("");
                setBaseUrl("");
                if (fileInputRef.current) fileInputRef.current.value = "";
              }}
            >
              Start another import
            </Button>
          </div>
          <Text
            as="h2"
            font="heading-h3"
          >{`New documents: ${batch.status}`}</Text>
          <Text font="secondary-body">{batch.stage.replaceAll("_", " ")}</Text>
          <Text font="main-ui-action">{`${batch.documents.length} new files · ${batch.documents.reduce((total, document) => total + document.chunk_count, 0)} added chunks`}</Text>
          {Boolean(batch.labeling_failed_count) && (
            <Text as="p" color="status-error-05">
              {`${batch.labeling_failed_count} chunks with failed labeling across this import`}
            </Text>
          )}
          <Text as="p" font="secondary-body" color="text-03">
            Review the added chunks below. Each card contains the full new
            content, labels and editing controls before approval.
          </Text>
          {batch.error && <Text color="status-error-05">{batch.error}</Text>}
          {batch.issues.map((issue, index) => (
            <Text key={index} color="status-error-05">
              {[
                issue.code.replaceAll("_", " "),
                issue.locator,
                issue.failure_detail,
              ]
                .filter(Boolean)
                .join(" · ")}
            </Text>
          ))}
          {pendingChunkEdits.length > 0 && (
            <div className="flex flex-col gap-2 rounded-08 border border-border-02 p-3">
              <Text font="main-ui-action">{`${pendingChunkEdits.length} unsaved chunk changes`}</Text>
              <Text font="secondary-body" color="text-03">
                Continue reviewing other chunks or pages. Save these changes
                together to start one labeling run.
              </Text>
              {chunkSaveError && (
                <Text color="status-error-05">{chunkSaveError}</Text>
              )}
              <div className="flex flex-wrap gap-2">
                <Button
                  disabled={
                    busy ||
                    Boolean(chunkSaveError) ||
                    lineageDirtyKeys.size > 0 ||
                    !["ready", "partial"].includes(batch.status)
                  }
                  onClick={() => void saveAllChunkChanges()}
                >
                  Save all chunk changes
                </Button>
                <Button
                  prominence="secondary"
                  disabled={busy}
                  onClick={() => {
                    setChunkDrafts({});
                    setDraftEpoch((current) => current + 1);
                    setRequestError(null);
                  }}
                >
                  Discard all chunk changes
                </Button>
              </div>
            </div>
          )}
          {batch.documents.map((document) => (
            <DocumentPreview
              key={document.id}
              batch={batch}
              document={document}
              navigation={navigation}
              busy={busy}
              dirtyKeys={dirtyKeys}
              onDirtyChange={onDirtyChange}
              chunkDrafts={chunkDrafts}
              draftEpoch={draftEpoch}
              onChunkDraftChange={onChunkDraftChange}
              onSaveChunk={(documentId, chunkId, draft, expectedReviewSha256) =>
                saveReview(
                  () =>
                    updateDocumentIntakeChunk(
                      batch.id,
                      documentId,
                      chunkId,
                      expectedReviewSha256,
                      draft
                    ),
                  true
                )
              }
              onSaveLineage={(documentId, selection, expectedReviewSha256) =>
                saveReview(() =>
                  saveDocumentIntakePredecessorSelection(
                    batch.id,
                    documentId,
                    expectedReviewSha256,
                    selection
                  )
                )
              }
              onNavigate={(documentId) => {
                if (busy) return;
                if (!batch.documents.some((item) => item.id === documentId))
                  return;
                setNavigation((current) => ({
                  documentId,
                  version: (current?.version ?? 0) + 1,
                }));
              }}
            />
          ))}
          {reprepareAvailable && (
            <section
              aria-label="Reprepare frozen source"
              className="flex flex-col gap-2 rounded-08 border border-border-02 p-3"
            >
              <Text font="main-ui-action">
                Reprepare chunks from frozen source
              </Text>
              <Text as="p" font="secondary-body" color="text-03">
                Uses the same new files, attachments and frozen converted text.
                Rebuilds their chunks and regenerates their labels with the
                current preparation rules. Existing PC Külliyatı records are
                preserved.
              </Text>
              <Text as="p" font="secondary-body" color="text-03">
                Available before approval when no saved manual changes or
                publication history need to be preserved. This does not correct
                the frozen source text.
              </Text>
              {!repreparePreview && !repreparePreviewError && (
                <Text font="secondary-body">
                  Checking preparation compatibility…
                </Text>
              )}
              {repreparePreview?.reason && (
                <Text as="p" color="status-error-05">
                  {repreparePreview.reason}
                </Text>
              )}
              {repreparePreviewError && (
                <Text as="p" color="status-error-05">
                  {errorMessage(repreparePreviewError)}
                </Text>
              )}
              {dirtyKeys.size > 0 && (
                <Text font="secondary-body" color="text-03">
                  Save or discard your changes before preparing again.
                </Text>
              )}
              <div>
                <Button
                  prominence="secondary"
                  disabled={
                    busy ||
                    dirtyKeys.size > 0 ||
                    !repreparePreview?.eligible ||
                    !repreparePreview.expected_sha256 ||
                    Boolean(repreparePreviewError)
                  }
                  onClick={() => void reprepareChunks()}
                >
                  Reprepare chunks from frozen source
                </Button>
              </div>
            </section>
          )}
          <div className="flex flex-wrap gap-2">
            {batchError && (
              <Button
                prominence="secondary"
                onClick={() => void refreshBatch()}
              >
                Refresh new document status
              </Button>
            )}
            {["partial", "blocked", "failed"].includes(batch.status) && (
              <Button
                disabled={busy || dirtyKeys.size > 0}
                prominence="secondary"
                onClick={() =>
                  void run(async (isCurrent) => {
                    const retried = await retryDocumentIntakeBatch(batch.id);
                    if (!isCurrent()) return;
                    setInitialBatch(retried);
                    await refreshBatch(retried, false);
                  })
                }
              >
                {batch.stage === "lineage"
                  ? "Retry document relationship"
                  : "Retry new documents"}
              </Button>
            )}
            <Button
              disabled={
                busy ||
                dirtyKeys.size > 0 ||
                batch.status !== "ready" ||
                !batch.review_sha256 ||
                Boolean(batchError)
              }
              onClick={() =>
                void run(async (isCurrent) => {
                  const approved = await approveDocumentIntakeBatch(
                    batch.id,
                    batch.review_sha256!
                  );
                  if (!isCurrent()) return;
                  setInitialBatch(approved);
                  await refreshBatch(approved, false);
                  await refreshHistory();
                  if (!isCurrent()) return;
                  toast.success(
                    "New document publication queued. Completion will appear after indexing succeeds."
                  );
                })
              }
            >
              Approve all new documents
            </Button>
          </div>
          {batch.status === "blocked" && batch.stage === "lineage" && (
            <Text as="p" font="secondary-body" color="text-04">
              Restore the previous file to PC Külliyatı, then retry the document
              relationship. Published chunks and labels will be kept.
            </Text>
          )}
          {dirtyKeys.size > 0 && (
            <Text as="p" font="secondary-body" color="text-04">
              Save or discard your changes before approving or leaving this
              review.
            </Text>
          )}
          {batch.status === "completed" && (
            <Text color="status-success-05">{`Import completed in ${batch.target_document_set_name}.`}</Text>
          )}
        </section>
      )}
    </div>
  );
}
