"use client";

import { useCallback, useState } from "react";

import { Button, InputTextArea, Modal, Tag, Text } from "@opal/components";
import { toast } from "@opal/layouts";
import { SvgEdit,
  SvgFileText,
} from "@opal/icons";
import { cn } from "@opal/utils";

import InputDatePicker from "@/refresh-components/inputs/InputDatePicker";
import InputTypeIn from "@/refresh-components/inputs/InputTypeIn";
import { useFileChunks } from "@/lib/regulatory/hooks";
import type {
  RegulatoryChunk,
  RegulatoryChunkUpdate,
} from "@/lib/regulatory/interfaces";
import {
  chunkPdfUrl,
  documentPdfUrl,
  patchChunk,
  renameUserFile,
} from "@/lib/regulatory/svc";
import type { ProjectFile } from "@/lib/projects/types";

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

function parseMetadataJson(value: string): Record<string, unknown> | null {
  const parsed: unknown = JSON.parse(value);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return null;
  }
  return parsed as Record<string, unknown>;
}

function parseHeadingPathJson(value: string): string[] | null {
  const parsed: unknown = JSON.parse(value);
  if (
    !Array.isArray(parsed) ||
    parsed.some((pathPart) => typeof pathPart !== "string")
  ) {
    return null;
  }
  return parsed as string[];
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

    let metadata: Record<string, unknown> | null;
    try {
      metadata = parseMetadataJson(metadataJson);
    } catch {
      metadata = null;
    }
    if (!metadata) {
      setError("Metadata must be a valid JSON object.");
      return;
    }

    let headingPath: string[] | null;
    try {
      headingPath = parseHeadingPathJson(headingPathJson);
    } catch {
      headingPath = null;
    }
    if (!headingPath) {
      setError("Heading path must be a JSON array of strings.");
      return;
    }
    if (!text.trim()) {
      setError("Chunk text cannot be empty.");
      return;
    }

    const update: RegulatoryChunkUpdate = {
      text,
      chunk_metadata: metadata,
      heading_path: headingPath,
      validity_start_date: toIsoDate(validityStart),
      clear_validity_start_date:
        validityStart === null && chunk.validity_start_date !== null,
      validity_end_date: toIsoDate(validityEnd),
      clear_validity_end_date:
        validityEnd === null && chunk.validity_end_date !== null,
    };

    setSaving(true);
    try {
      await patchChunk(chunk.id, update);
      toast.success("Chunk updated and re-indexed.");
      onSaved();
      onClose();
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "Update failed."
      );
    } finally {
      setSaving(false);
    }
  }, [
    chunk,
    headingPathJson,
    metadataJson,
    onClose,
    onSaved,
    text,
    validityEnd,
    validityStart,
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
                onChange={(event) => setText(event.target.value)}
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
                onChange={(event) => setHeadingPathJson(event.target.value)}
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
                onChange={(event) => setMetadataJson(event.target.value)}
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

interface RegulatoryFileDetailsProps {
  file: ProjectFile;
  onFileRenamed: (name: string) => void;
}

export default function RegulatoryFileDetails({
  file,
  onFileRenamed,
}: RegulatoryFileDetailsProps) {
  const { chunks, error, isLoading, refreshChunks } = useFileChunks(file.id);
  const [editingChunk, setEditingChunk] = useState<RegulatoryChunk | null>(
    null
  );
  const [expandedJson, setExpandedJson] = useState<Set<string>>(new Set());
  const [renaming, setRenaming] = useState(false);
  const [newFileName, setNewFileName] = useState(file.name);

  const toggleJson = useCallback((chunkId: string) => {
    setExpandedJson((currentExpandedChunks) => {
      const nextExpandedChunks = new Set(currentExpandedChunks);
      if (nextExpandedChunks.has(chunkId)) {
        nextExpandedChunks.delete(chunkId);
      } else {
        nextExpandedChunks.add(chunkId);
      }
      return nextExpandedChunks;
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
    } catch (requestError) {
      toast.error(
        requestError instanceof Error ? requestError.message : "Rename failed."
      );
    } finally {
      setRenaming(false);
    }
  }, [file.id, file.name, newFileName, onFileRenamed]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        {renaming ? (
          <>
            <InputTypeIn
              value={newFileName}
              onChange={(event) => setNewFileName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void handleRename();
                if (event.key === "Escape") setRenaming(false);
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
              tooltip="Rename file"
              onClick={() => {
                setNewFileName(file.name);
                setRenaming(true);
              }}
            />
          </>
        )}
        {!isLoading && (
          <Text font="secondary-body" color="text-03">
            {`${chunks.length} chunks`}
          </Text>
        )}
        <Button
          prominence="secondary"
          icon={SvgFileText}
          size="sm"
          href={documentPdfUrl(file.id)}
          target="_blank"
          tooltip="The document as uploaded, before chunking"
        >
          Document PDF
        </Button>
      </div>

      {isLoading ? (
        <Text font="main-ui-body" color="text-03">
          Loading chunks…
        </Text>
      ) : error ? (
        <Text font="main-ui-body" color="text-05">
          Failed to load chunks for this file.
        </Text>
      ) : chunks.length === 0 ? (
        <Text font="main-ui-body" color="text-03">
          No regulatory chunks are available for this file yet.
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
                  size="sm"
                  href={chunkPdfUrl(chunk.id)}
                  target="_blank"
                  tooltip="This chunk on its own"
                >
                  Chunk PDF
                </Button>
                <Button
                  prominence="tertiary"
                  icon={SvgEdit}
                  size="sm"
                  tooltip="Edit chunk"
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
