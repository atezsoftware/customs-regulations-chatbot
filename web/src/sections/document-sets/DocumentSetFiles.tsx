"use client";

import { useCallback, useRef, useState } from "react";
import useSWR from "swr";

import { Button, LineItemButton, Tag, Text } from "@opal/components";
import { toast } from "@opal/layouts";
import {
  SvgChevronLeft,
  SvgFileText,
  SvgFolderIn,
  SvgFolderPlus,
  SvgMinusCircle,
  SvgPlus,
  SvgSearchMenu,
} from "@opal/icons";

import type {
  ProjectFile,
  RegulatoryIndexingProgress,
} from "@/lib/projects/types";
import { UserFileStatus } from "@/lib/projects/types";
import { useSettings } from "@/lib/settings/hooks";
import { documentPdfUrl } from "@/lib/regulatory/svc";
import {
  getDocumentSetFiles,
  getDocumentSetFilesKey,
  indexDocumentSetChunkedFiles,
  indexDocumentSetFile,
  unlinkFileFromDocumentSet,
  uploadDocumentSetFiles,
} from "@/app/admin/documents/sets/lib";
import RegulatoryFileDetails from "@/sections/document-sets/RegulatoryFileDetails";

const ACTIVE_FILE_STATUSES = new Set<UserFileStatus>([
  UserFileStatus.UPLOADING,
  UserFileStatus.PROCESSING,
  UserFileStatus.INDEXING,
]);

const INDEXING_STAGE_LABELS: Record<
  RegulatoryIndexingProgress["stage"],
  string
> = {
  PREPARING: "Dizinleme hazırlanıyor",
  CONTEXT_SUBMIT: "Bağlam işi gönderiliyor",
  CONTEXT_WAIT: "Bağlam işi bekleniyor",
  CONTEXT_APPLY: "Bağlam sonuçları işleniyor",
  EMBEDDING: "Vektörler oluşturuluyor",
  INDEX_WRITE: "Arama dizinine yazılıyor",
  VERIFY: "Dizin doğrulanıyor",
  PUBLISH: "Dizin yayımlanıyor",
};

const PROVIDER_BATCH_STATE_LABELS: Record<string, string> = {
  SUBMITTING: "gönderiliyor",
  RECONCILE_REQUIRED: "sağlayıcı kontrolü bekliyor",
  RECONCILED_ABSENT: "sağlayıcıda bulunamadı",
  MANUAL_RECONCILE_REQUIRED: "operatör kontrolü gerekiyor",
  SUBMITTED: "gönderildi",
};

function getProviderBatchDescription(value: string | null): string | null {
  if (!value) return null;
  const [provider, state] = value.split(":", 2);
  if (!provider || !state) return null;
  const stateLabel = PROVIDER_BATCH_STATE_LABELS[state];
  if (!stateLabel) return null;
  const providerLabel =
    provider === "openrouter"
      ? "OpenRouter"
      : provider === "vertex"
        ? "Vertex"
        : null;
  return providerLabel ? `${providerLabel} toplu işi ${stateLabel}` : null;
}

function getIndexingProgressDescription(
  progress: RegulatoryIndexingProgress
): string {
  const details = [INDEXING_STAGE_LABELS[progress.stage]];
  const hasReliableCounts =
    Number.isInteger(progress.total_items) &&
    Number.isInteger(progress.completed_items) &&
    progress.total_items > 0 &&
    progress.completed_items >= 0 &&
    progress.completed_items <= progress.total_items;
  if (hasReliableCounts) {
    const percentage = Math.floor(
      (progress.completed_items / progress.total_items) * 100
    );
    details.push(
      `${progress.completed_items}/${progress.total_items} (${percentage}%)`
    );
  }
  if (progress.attempt_count > 0) {
    details.push(`Deneme ${progress.attempt_count}`);
  }
  if (progress.next_retry_at) {
    details.push("Yeniden deneme planlandı");
  }
  const providerBatchDescription = getProviderBatchDescription(
    progress.provider_batch_state
  );
  if (providerBatchDescription) {
    details.push(providerBatchDescription);
  }
  if (progress.error_summary) {
    details.push(progress.error_summary);
  }
  return details.join(" · ");
}

function getFileStatusDescription(file: ProjectFile): string {
  const normalizedStatus = file.status.toLowerCase().replaceAll("_", " ");
  const status =
    normalizedStatus.length > 0
      ? normalizedStatus[0]!.toUpperCase() + normalizedStatus.slice(1)
      : "Unknown";
  const details = [status];
  if (file.chunk_count != null) {
    details.push(`${file.chunk_count} chunks`);
  }
  if (file.regulatory_indexing_progress) {
    details.push(
      getIndexingProgressDescription(file.regulatory_indexing_progress)
    );
  }
  return details.join(" · ");
}

// Extensions a markdown-only deployment accepts. Mirrors
// MARKDOWN_UPLOAD_EXTENSIONS in backend/onyx/file_processing/import_capability.py.
const MARKDOWN_ONLY_ACCEPT = ".md,.mdx,.zip";

interface UploadControlsProps {
  disabled: boolean;
  markdownOnly: boolean;
  onUpload: (files: File[]) => Promise<void>;
}

function UploadControls({
  disabled,
  markdownOnly,
  onUpload,
}: UploadControlsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const archiveInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList?.length) return;
      await onUpload(Array.from(fileList));
    },
    [onUpload]
  );

  return (
    <div className="flex flex-wrap items-center gap-2">
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        {...(markdownOnly ? { accept: MARKDOWN_ONLY_ACCEPT } : {})}
        aria-label="Upload files to document set"
        onChange={(event) => {
          void handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        className="hidden"
        {...(markdownOnly ? { accept: MARKDOWN_ONLY_ACCEPT } : {})}
        aria-label="Upload folder to document set"
        {...{ webkitdirectory: "", directory: "" }}
        onChange={(event) => {
          void handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <Button
        icon={SvgPlus}
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled}
      >
        Upload Files
      </Button>
      <input
        ref={archiveInputRef}
        type="file"
        accept=".zip"
        className="hidden"
        aria-label="Upload archive to document set"
        onChange={(event) => {
          void handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <Button
        icon={SvgFolderPlus}
        prominence="secondary"
        onClick={() => folderInputRef.current?.click()}
        disabled={disabled}
      >
        Upload Folder
      </Button>
      <Button
        icon={SvgFolderIn}
        prominence="secondary"
        onClick={() => archiveInputRef.current?.click()}
        disabled={disabled}
        tooltip="Upload a .zip; every document inside is indexed separately"
      >
        Upload Archive
      </Button>
      {disabled && (
        <Text font="main-ui-body" color="text-03">
          Uploading…
        </Text>
      )}
    </div>
  );
}

interface DocumentSetFilesProps {
  documentSetId: number;
  documentSetName: string;
}

export default function DocumentSetFiles({
  documentSetId,
  documentSetName,
}: DocumentSetFilesProps) {
  const {
    document_import_enabled: documentImportEnabled = true,
    markdown_import_enabled: markdownImportEnabled = true,
  } = useSettings();
  // Full document import implies markdown import; the reverse does not hold.
  const canUpload = documentImportEnabled || markdownImportEnabled;
  const [selectedFile, setSelectedFile] = useState<ProjectFile | null>(null);
  const [uploading, setUploading] = useState(false);
  const [removingFileId, setRemovingFileId] = useState<string | null>(null);
  const [indexingFileId, setIndexingFileId] = useState<string | null>(null);
  const [indexingAll, setIndexingAll] = useState(false);

  const {
    data: files,
    error,
    isLoading,
    mutate: refreshFiles,
  } = useSWR<ProjectFile[]>(
    getDocumentSetFilesKey(documentSetId),
    () => getDocumentSetFiles(documentSetId),
    {
      refreshInterval: (latestFiles) =>
        latestFiles?.some((file) => ACTIVE_FILE_STATUSES.has(file.status))
          ? 4000
          : 0,
    }
  );

  const handleUpload = useCallback(
    async (uploadList: File[]) => {
      setUploading(true);
      try {
        const result = await uploadDocumentSetFiles(documentSetId, uploadList);
        if (result.rejected_files.length > 0) {
          const firstRejection = result.rejected_files[0];
          toast.error(
            `${result.rejected_files.length} file(s) rejected${
              firstRejection
                ? `: ${firstRejection.file_name} — ${firstRejection.reason}`
                : "."
            }`
          );
        }
        if (result.user_files.length > 0) {
          toast.success(`${result.user_files.length} file(s) uploading.`);
        }
        await refreshFiles();
      } catch (requestError) {
        toast.error(
          requestError instanceof Error
            ? requestError.message
            : "Upload failed."
        );
      } finally {
        setUploading(false);
      }
    },
    [documentSetId, refreshFiles]
  );

  const chunkedFiles = files?.filter(
    (file) => file.status === UserFileStatus.CHUNKED
  );

  const handleIndexFile = useCallback(
    async (file: ProjectFile) => {
      setIndexingFileId(file.id);
      try {
        await indexDocumentSetFile(documentSetId, file.id);
        toast.success(`Indexing ${file.name}.`);
        await refreshFiles();
      } catch (requestError) {
        toast.error(
          requestError instanceof Error
            ? requestError.message
            : "Indexing failed."
        );
      } finally {
        setIndexingFileId(null);
      }
    },
    [documentSetId, refreshFiles]
  );

  const handleIndexAllChunked = useCallback(async () => {
    setIndexingAll(true);
    try {
      const { queued } = await indexDocumentSetChunkedFiles(documentSetId);
      toast.success(`Indexing ${queued} file(s).`);
      await refreshFiles();
    } catch (requestError) {
      toast.error(
        requestError instanceof Error
          ? requestError.message
          : "Indexing failed."
      );
    } finally {
      setIndexingAll(false);
    }
  }, [documentSetId, refreshFiles]);

  const handleRemoveFile = useCallback(
    async (file: ProjectFile) => {
      setRemovingFileId(file.id);
      try {
        await unlinkFileFromDocumentSet(documentSetId, file.id);
        toast.success(`Removed ${file.name} from this document set.`);
        await refreshFiles();
      } catch (requestError) {
        toast.error(
          requestError instanceof Error
            ? requestError.message
            : "Remove failed."
        );
      } finally {
        setRemovingFileId(null);
      }
    },
    [documentSetId, refreshFiles]
  );

  if (selectedFile) {
    return (
      <div className="flex flex-col gap-4">
        <div>
          <Button
            prominence="tertiary"
            icon={SvgChevronLeft}
            onClick={() => setSelectedFile(null)}
          >
            Back to files
          </Button>
        </div>
        <Text font="main-ui-body" color="text-03" as="p">
          Inspect and edit the regulatory chunks generated from this file.
        </Text>
        <RegulatoryFileDetails
          file={selectedFile}
          onFileRenamed={(name) => {
            setSelectedFile((currentFile) =>
              currentFile ? { ...currentFile, name } : null
            );
            void refreshFiles();
          }}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Text font="main-ui-body" color="text-03" as="p">
        {`Upload regulatory documents to ${documentSetName}. Each file is chunked along its structure and indexed as part of this document set.`}
      </Text>

      {canUpload ? (
        <>
          <UploadControls
            disabled={uploading}
            markdownOnly={!documentImportEnabled}
            onUpload={handleUpload}
          />
          {!documentImportEnabled && (
            <Text font="main-ui-body" color="text-03" as="p">
              This deployment accepts only Markdown documents and .zip archives
              of them. Other formats are converted by the separate importer
              deployment.
            </Text>
          )}
        </>
      ) : (
        <Text font="main-ui-body" color="text-03" as="p">
          Document import runs in the separate importer deployment. Existing
          chunks remain available here for inspection and search.
        </Text>
      )}

      {isLoading ? (
        <Text font="main-ui-body" color="text-03">
          Loading files…
        </Text>
      ) : error ? (
        <Text font="main-ui-body" color="text-05">
          Failed to load files for this document set.
        </Text>
      ) : !files?.length ? (
        <div className="rounded-lg border border-dashed border-border-01 p-6">
          <Text font="main-ui-body" color="text-03" as="p">
            No files have been uploaded to this document set yet.
          </Text>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {chunkedFiles && chunkedFiles.length > 0 && (
            <div className="flex items-center gap-2">
              <Button
                icon={SvgSearchMenu}
                prominence="secondary"
                disabled={indexingAll}
                onClick={() => void handleIndexAllChunked()}
              >
                {`Index all chunked (${chunkedFiles.length})`}
              </Button>
              <Text font="main-ui-body" color="text-03">
                Chunks are ready to review; indexing makes them searchable.
              </Text>
            </div>
          )}
          {files.map((file) => (
            <LineItemButton
              key={file.id}
              sizePreset="main-ui"
              variant="section"
              icon={SvgFileText}
              title={file.name}
              description={getFileStatusDescription(file)}
              onClick={() => setSelectedFile(file)}
              rightChildren={
                <div
                  className="flex items-center gap-2"
                  onClick={(event) => event.stopPropagation()}
                >
                  {ACTIVE_FILE_STATUSES.has(file.status) && (
                    <Tag title="Processing" />
                  )}
                  <Button
                    prominence="tertiary"
                    icon={SvgFileText}
                    size="sm"
                    href={documentPdfUrl(file.id)}
                    target="_blank"
                    aria-label={`Open ${file.name} as PDF in a new tab`}
                    tooltip="Open the complete document as PDF"
                  >
                    PDF
                  </Button>
                  {file.status === UserFileStatus.CHUNKED && (
                    <Button
                      prominence="secondary"
                      size="sm"
                      disabled={indexingFileId === file.id}
                      onClick={() => void handleIndexFile(file)}
                    >
                      Index
                    </Button>
                  )}
                  <Button
                    variant="danger"
                    prominence="tertiary"
                    icon={SvgMinusCircle}
                    size="sm"
                    tooltip="Remove from document set"
                    disabled={removingFileId === file.id}
                    onClick={() => void handleRemoveFile(file)}
                  />
                </div>
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
