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

import type { ProjectFile } from "@/lib/projects/types";
import { UserFileStatus } from "@/lib/projects/types";
import { useSettings } from "@/lib/settings/hooks";
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

function getFileStatusDescription(file: ProjectFile): string {
  const normalizedStatus = file.status.toLowerCase().replaceAll("_", " ");
  const status =
    normalizedStatus.length > 0
      ? normalizedStatus[0]!.toUpperCase() + normalizedStatus.slice(1)
      : "Unknown";
  return file.chunk_count == null
    ? status
    : `${status} · ${file.chunk_count} chunks`;
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
        requestError instanceof Error ? requestError.message : "Indexing failed."
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
