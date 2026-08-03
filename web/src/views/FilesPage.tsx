"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import useSWR from "swr";

import { Button, InputTextArea, Modal, Tag, Text } from "@opal/components";
import { Content, ContentAction, SettingsLayouts, toast } from "@opal/layouts";
import { cn } from "@opal/utils";
import InputTypeIn from "@/refresh-components/inputs/InputTypeIn";
import InputDatePicker from "@/refresh-components/inputs/InputDatePicker";
import {
  SvgChevronLeft,
  SvgEdit,
  SvgFileText,
  SvgFolder,
  SvgFolderPlus,
  SvgPlus,
  SvgTrash,
} from "@opal/icons";
import { useProjects } from "@/lib/projects/hooks";
import {
  createProject,
  deleteUserFile,
  getFilesInProject,
  uploadFiles,
} from "@/lib/projects/svc";
import type { Project, ProjectFile } from "@/lib/projects/types";
import { useFileChunks } from "@/lib/regulatory/hooks";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import { patchChunk, renameUserFile } from "@/lib/regulatory/svc";
import { useSettings } from "@/lib/settings/hooks";
import { useUser } from "@/providers/UserProvider";

const ACTIVE_FILE_STATUSES = ["PROCESSING", "INDEXING"];

function parseIsoDate(value: string | null): Date | null {
  if (!value) return null;
  const parsed = new Date(`${value}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function toIsoDate(date: Date | null): string | undefined {
  if (!date) return undefined;
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

interface DirectoryListProps {
  projects: Project[];
  onSelect: (project: Project) => void;
  onCreate: (name: string) => Promise<void>;
}

function DirectoryList({ projects, onSelect, onCreate }: DirectoryListProps) {
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  const handleCreate = useCallback(async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      await onCreate(name);
      setNewName("");
    } finally {
      setCreating(false);
    }
  }, [newName, onCreate]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2 items-center">
        <InputTypeIn
          placeholder="New directory name..."
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void handleCreate();
          }}
        />
        <Button
          icon={SvgFolderPlus}
          onClick={() => void handleCreate()}
          disabled={creating || !newName.trim()}
        >
          Create
        </Button>
      </div>

      <div className="flex flex-col gap-2">
        {projects.length === 0 && (
          <Text font="main-ui-body" color="text-03">
            No directories yet. Create one to start uploading files.
          </Text>
        )}
        {projects.map((project) => (
          <button
            key={project.id}
            type="button"
            className="text-left rounded-lg border border-border-02 hover:bg-background-tint-02 p-3"
            onClick={() => onSelect(project)}
          >
            <Content
              sizePreset="main-ui"
              variant="section"
              icon={SvgFolder}
              title={project.name}
              description={new Date(project.created_at).toLocaleDateString()}
            />
          </button>
        ))}
      </div>
    </div>
  );
}

interface FileListProps {
  project: Project;
  files: ProjectFile[];
  isLoading: boolean;
  onSelectFile: (file: ProjectFile) => void;
  onUpload: (files: File[]) => Promise<void>;
  onDeleteFile: (file: ProjectFile) => Promise<void>;
  documentImportEnabled: boolean;
}

function FileList({
  project,
  files,
  isLoading,
  onSelectFile,
  onUpload,
  onDeleteFile,
  documentImportEnabled,
}: FileListProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const handleFiles = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0) return;
      setUploading(true);
      try {
        await onUpload(Array.from(fileList));
      } finally {
        setUploading(false);
      }
    },
    [onUpload]
  );

  return (
    <div className="flex flex-col gap-4">
      {documentImportEnabled ? (
        <div className="flex gap-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              void handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={folderInputRef}
            type="file"
            multiple
            className="hidden"
            // Non-standard but universally supported attribute enabling
            // directory selection; typed via spread since React's TS defs
            // lag behind.
            {...{ webkitdirectory: "", directory: "" }}
            onChange={(e) => {
              void handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <Button
            icon={SvgPlus}
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            Upload Files
          </Button>
          <Button
            icon={SvgFolderPlus}
            prominence="secondary"
            onClick={() => folderInputRef.current?.click()}
            disabled={uploading}
          >
            Upload Folder
          </Button>
          {uploading && (
            <Text font="main-ui-body" color="text-03">
              Uploading…
            </Text>
          )}
        </div>
      ) : (
        <Text font="main-ui-body" color="text-03">
          Document import runs in the separate importer deployment. Existing
          chunks remain available here for inspection and search.
        </Text>
      )}

      {isLoading ? (
        <Text font="main-ui-body" color="text-03">
          Loading files…
        </Text>
      ) : files.length === 0 ? (
        <Text font="main-ui-body" color="text-03">
          {`No files in ${project.name} yet.`}
        </Text>
      ) : (
        <div className="flex flex-col gap-2">
          {files.map((file) => (
            <div
              key={file.id}
              className="rounded-lg border border-border-02 hover:bg-background-tint-02 p-3 cursor-pointer"
              onClick={() => onSelectFile(file)}
            >
              <ContentAction
                sizePreset="main-ui"
                variant="section"
                icon={SvgFileText}
                title={file.name}
                description={`${file.status}${
                  file.chunk_count != null
                    ? ` — ${file.chunk_count} chunks`
                    : ""
                }`}
                rightChildren={
                  <div className="flex items-center gap-2">
                    {ACTIVE_FILE_STATUSES.includes(file.status) && (
                      <Tag title="Processing" />
                    )}
                    <Button
                      variant="danger"
                      prominence="tertiary"
                      icon={SvgTrash}
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        void onDeleteFile(file);
                      }}
                    />
                  </div>
                }
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface ChunkEditModalProps {
  chunk: RegulatoryChunk;
  onClose: () => void;
  onSaved: () => void;
}

function ChunkEditModal({ chunk, onClose, onSaved }: ChunkEditModalProps) {
  const [text, setText] = useState(chunk.text);
  const [metadataJson, setMetadataJson] = useState(
    JSON.stringify(chunk.chunk_metadata, null, 2)
  );
  const [headingPathJson, setHeadingPathJson] = useState(
    JSON.stringify(chunk.heading_path, null, 2)
  );
  const [validityStart, setValidityStart] = useState<Date | null>(
    parseIsoDate(chunk.validity_start_date)
  );
  const [validityEnd, setValidityEnd] = useState<Date | null>(
    parseIsoDate(chunk.validity_end_date)
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = useCallback(async () => {
    setError(null);

    let metadata: Record<string, unknown>;
    let headingPath: string[];
    try {
      metadata = JSON.parse(metadataJson);
    } catch {
      setError("Metadata is not valid JSON.");
      return;
    }
    try {
      headingPath = JSON.parse(headingPathJson);
      if (
        !Array.isArray(headingPath) ||
        headingPath.some((part) => typeof part !== "string")
      ) {
        throw new Error("not a string array");
      }
    } catch {
      setError("Heading path must be a JSON array of strings.");
      return;
    }
    if (!text.trim()) {
      setError("Chunk text cannot be empty.");
      return;
    }

    setSaving(true);
    try {
      await patchChunk(chunk.id, {
        text,
        chunk_metadata: metadata,
        heading_path: headingPath,
        validity_start_date: toIsoDate(validityStart),
        clear_validity_start_date:
          validityStart === null && chunk.validity_start_date !== null,
        validity_end_date: toIsoDate(validityEnd),
        clear_validity_end_date:
          validityEnd === null && chunk.validity_end_date !== null,
      });
      toast.success("Chunk updated and re-indexed.");
      onSaved();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Update failed.");
    } finally {
      setSaving(false);
    }
  }, [
    chunk,
    text,
    metadataJson,
    headingPathJson,
    validityStart,
    validityEnd,
    onClose,
    onSaved,
  ]);

  return (
    <Modal open onOpenChange={(open: boolean) => !open && onClose()}>
      <Modal.Content width="md">
        <Modal.Header
          title={`Edit Chunk #${chunk.position}`}
          description={chunk.heading_path.join(" > ")}
        />
        <Modal.Body>
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <Text font="main-ui-action" color="text-04">
                Text
              </Text>
              <InputTextArea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                autoResize
                maxRows={14}
              />
            </div>

            <div className="flex gap-4">
              <div className="flex flex-col gap-1 flex-1">
                <Text font="main-ui-action" color="text-04">
                  Validity Start
                </Text>
                <InputDatePicker
                  selectedDate={validityStart}
                  setSelectedDate={setValidityStart}
                />
              </div>
              <div className="flex flex-col gap-1 flex-1">
                <Text font="main-ui-action" color="text-04">
                  Validity End
                </Text>
                <InputDatePicker
                  selectedDate={validityEnd}
                  setSelectedDate={setValidityEnd}
                />
              </div>
            </div>

            <div className="flex flex-col gap-1">
              <Text font="main-ui-action" color="text-04">
                Heading Path (JSON)
              </Text>
              <InputTextArea
                value={headingPathJson}
                onChange={(e) => setHeadingPathJson(e.target.value)}
                rows={3}
                autoResize
                maxRows={8}
              />
            </div>

            <div className="flex flex-col gap-1">
              <Text font="main-ui-action" color="text-04">
                Metadata (JSON)
              </Text>
              <InputTextArea
                value={metadataJson}
                onChange={(e) => setMetadataJson(e.target.value)}
                rows={6}
                autoResize
                maxRows={14}
              />
            </div>

            {error && (
              <Text font="main-ui-body" color="text-05" as="p">
                {error}
              </Text>
            )}
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button prominence="secondary" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={() => void handleSave()} disabled={saving}>
            {saving ? "Saving…" : "Save & Re-index"}
          </Button>
        </Modal.Footer>
      </Modal.Content>
    </Modal>
  );
}

interface ChunkListProps {
  file: ProjectFile;
  onFileRenamed: (name: string) => void;
}

function ChunkList({ file, onFileRenamed }: ChunkListProps) {
  const { chunks, isLoading, refreshChunks } = useFileChunks(file.id);
  const [editingChunk, setEditingChunk] = useState<RegulatoryChunk | null>(
    null
  );
  const [expandedJson, setExpandedJson] = useState<Set<string>>(new Set());
  const [renaming, setRenaming] = useState(false);
  const [newFileName, setNewFileName] = useState(file.name);

  const toggleJson = useCallback((chunkId: string) => {
    setExpandedJson((prev) => {
      const next = new Set(prev);
      if (next.has(chunkId)) {
        next.delete(chunkId);
      } else {
        next.add(chunkId);
      }
      return next;
    });
  }, []);

  const handleRename = useCallback(async () => {
    const name = newFileName.trim();
    if (!name || name === file.name) {
      setRenaming(false);
      return;
    }
    try {
      await renameUserFile(file.id, name);
      toast.success("File renamed.");
      onFileRenamed(name);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Rename failed.");
    } finally {
      setRenaming(false);
    }
  }, [file, newFileName, onFileRenamed]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        {renaming ? (
          <>
            <InputTypeIn
              value={newFileName}
              onChange={(e) => setNewFileName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleRename();
                if (e.key === "Escape") setRenaming(false);
              }}
            />
            <Button size="sm" onClick={() => void handleRename()}>
              Save
            </Button>
          </>
        ) : (
          <>
            <Text font="heading-h3" color="text-05">
              {file.name}
            </Text>
            <Button
              prominence="tertiary"
              icon={SvgEdit}
              size="sm"
              onClick={() => {
                setNewFileName(file.name);
                setRenaming(true);
              }}
            />
          </>
        )}
        <Text font="secondary-body" color="text-03">
          {`${chunks.length} chunks`}
        </Text>
      </div>

      {isLoading ? (
        <Text font="main-ui-body" color="text-03">
          Loading chunks…
        </Text>
      ) : (
        chunks.map((chunk) => (
          <div
            key={chunk.id}
            className={cn(
              "rounded-lg border p-3 flex flex-col gap-2",
              chunk.status === "superseded"
                ? "border-border-01 bg-background-tint-01"
                : "border-border-02"
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <Text font="secondary-body" color="text-03">
                {`#${chunk.position} · ${chunk.chunk_type ?? "text"} · ${chunk.heading_path.join(" > ")}`}
              </Text>
              <div className="flex items-center gap-1 shrink-0">
                {chunk.source === "amendment" && <Tag title="Amendment" />}
                {chunk.status === "superseded" && <Tag title="Superseded" />}
                <Button
                  prominence="tertiary"
                  size="sm"
                  onClick={() => toggleJson(chunk.id)}
                >
                  JSON
                </Button>
                <Button
                  prominence="tertiary"
                  icon={SvgEdit}
                  size="sm"
                  onClick={() => setEditingChunk(chunk)}
                />
              </div>
            </div>

            <Text font="main-ui-body" color="text-05" as="p">
              {chunk.text}
            </Text>

            {(chunk.validity_start_date || chunk.validity_end_date) && (
              <Text font="secondary-body" color="text-03">
                {`Validity: ${chunk.validity_start_date ?? "—"} → ${chunk.validity_end_date ?? "open"}`}
              </Text>
            )}

            {expandedJson.has(chunk.id) && (
              <pre className="text-xs bg-background-tint-02 rounded-md p-3 overflow-x-auto">
                {JSON.stringify(chunk, null, 2)}
              </pre>
            )}
          </div>
        ))
      )}

      {editingChunk && (
        <ChunkEditModal
          chunk={editingChunk}
          onClose={() => setEditingChunk(null)}
          onSaved={() => void refreshChunks()}
        />
      )}
    </div>
  );
}

function AdminFilesPage() {
  const { projects, refreshProjects } = useProjects();
  const { document_import_enabled: documentImportEnabled = true } =
    useSettings();
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [selectedFile, setSelectedFile] = useState<ProjectFile | null>(null);

  const {
    data: files,
    isLoading: filesLoading,
    mutate: refreshFiles,
  } = useSWR<ProjectFile[]>(
    selectedProject ? `project-files-${selectedProject.id}` : null,
    () => getFilesInProject(selectedProject!.id),
    {
      // Poll while any file is still being processed so statuses/chunk
      // counts update without a manual refresh.
      refreshInterval: (latest) =>
        latest?.some((f) => ACTIVE_FILE_STATUSES.includes(f.status)) ? 4000 : 0,
    }
  );

  const handleCreateDirectory = useCallback(
    async (name: string) => {
      try {
        await createProject(name);
        await refreshProjects();
        toast.success(`Directory "${name}" created.`);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Create failed.");
      }
    },
    [refreshProjects]
  );

  const handleUpload = useCallback(
    async (uploadList: File[]) => {
      if (!selectedProject) return;
      try {
        const result = await uploadFiles(uploadList, selectedProject.id);
        if (result.rejected_files?.length) {
          toast.error(`${result.rejected_files.length} file(s) were rejected.`);
        } else {
          toast.success(`${uploadList.length} file(s) uploading.`);
        }
        await refreshFiles();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Upload failed.");
      }
    },
    [selectedProject, refreshFiles]
  );

  const handleDeleteFile = useCallback(
    async (file: ProjectFile) => {
      try {
        await deleteUserFile(file.id);
        toast.success(`Deleted ${file.name}.`);
        await refreshFiles();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Delete failed.");
      }
    },
    [refreshFiles]
  );

  const headerTitle = useMemo(() => {
    if (selectedFile) return selectedFile.name;
    if (selectedProject) return selectedProject.name;
    return "Files";
  }, [selectedFile, selectedProject]);

  const backButton = useMemo(() => {
    if (!selectedProject) return null;
    return (
      <Button
        prominence="tertiary"
        icon={SvgChevronLeft}
        onClick={() => {
          if (selectedFile) {
            setSelectedFile(null);
          } else {
            setSelectedProject(null);
          }
        }}
      >
        Back
      </Button>
    );
  }, [selectedProject, selectedFile]);

  return (
    <SettingsLayouts.Root width="lg">
      <SettingsLayouts.Header
        icon={SvgFolder}
        title={headerTitle}
        description={
          selectedFile
            ? "Inspect and edit the chunks generated from this file."
            : selectedProject
              ? "Files in this directory. Click a file to view its chunks."
              : "Directories of regulatory documents. Every file is chunked along its structure (madde / fıkra / bent) and indexed for search."
        }
        rightChildren={backButton}
      />
      <SettingsLayouts.Body>
        {selectedFile ? (
          <ChunkList
            file={selectedFile}
            onFileRenamed={(name) => {
              setSelectedFile({ ...selectedFile, name });
              void refreshFiles();
            }}
          />
        ) : selectedProject ? (
          <FileList
            project={selectedProject}
            files={files ?? []}
            isLoading={filesLoading}
            onSelectFile={setSelectedFile}
            onUpload={handleUpload}
            onDeleteFile={handleDeleteFile}
            documentImportEnabled={documentImportEnabled}
          />
        ) : (
          <DirectoryList
            projects={projects ?? []}
            onSelect={setSelectedProject}
            onCreate={handleCreateDirectory}
          />
        )}
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}

export default function FilesPage() {
  const router = useRouter();
  const { user, isAdmin } = useUser();

  useEffect(() => {
    if (user && !isAdmin) {
      router.replace("/app");
    }
  }, [isAdmin, router, user]);

  if (!user || !isAdmin) return null;
  return <AdminFilesPage />;
}
