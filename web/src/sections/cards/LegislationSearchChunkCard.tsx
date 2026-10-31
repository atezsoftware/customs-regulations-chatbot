"use client";

import { useState } from "react";
import { Button, Tag, Text } from "@opal/components";
import { SvgEdit, SvgFileText } from "@opal/icons";
import type { LegislationSearchResult } from "@/lib/legistations/interfaces";
import { chunkEditRestriction } from "@/lib/regulatory/chunkEditing";
import { ChunkEditModal } from "@/sections/document-sets/RegulatoryFileDetails";

interface LegislationSearchChunkCardProps {
  result: LegislationSearchResult;
  onOpenFile: () => void;
  onSaved: () => void;
}

export default function LegislationSearchChunkCard({
  result,
  onOpenFile,
  onSaved,
}: LegislationSearchChunkCardProps) {
  const [details, setDetails] = useState(false);
  const [editing, setEditing] = useState(false);
  const { chunk, file, score } = result;
  const editRestriction = chunkEditRestriction(chunk);
  return (
    <article className="flex flex-col gap-3 rounded-lg border border-border-02 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0 flex-1 flex flex-col gap-1 break-words [overflow-wrap:anywhere]">
          <Text font="main-ui-action" as="h3">
            {file.name}
          </Text>
          <Text
            font="secondary-body"
            color="text-03"
          >{`#${chunk.position} · ${chunk.chunk_type ?? "text"} · ${chunk.heading_path.join(" > ")}`}</Text>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {chunk.status === "superseded" && <Tag title="Önceki sürüm" />}
          {file.pending_review && <Tag title="İnceleme bekliyor" />}
          <Button
            prominence="secondary"
            size="sm"
            icon={SvgFileText}
            onClick={onOpenFile}
          >
            Dosyaya git
          </Button>
          <Button
            prominence="tertiary"
            size="sm"
            icon={SvgEdit}
            disabled={file.pending_review || editRestriction !== null}
            tooltip={editRestriction ?? undefined}
            onClick={() => setEditing(true)}
          >
            Düzenle
          </Button>
        </div>
      </div>
      {editRestriction && (
        <Text font="secondary-body" color="text-03">
          {editRestriction}
        </Text>
      )}
      <div className="whitespace-pre-wrap break-words">
        <Text font="main-ui-body" as="p">
          {chunk.text}
        </Text>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button
          prominence="tertiary"
          size="sm"
          aria-expanded={details}
          onClick={() => setDetails(!details)}
        >
          Chunk detayları
        </Button>
        {score !== null && (
          <Text
            font="secondary-body"
            color="text-03"
          >{`Skor: ${score.toFixed(4)}`}</Text>
        )}
        {(chunk.validity_start_date || chunk.validity_end_date) && (
          <Text
            font="secondary-body"
            color="text-03"
          >{`${chunk.validity_start_date ?? "—"} → ${chunk.validity_end_date ?? "Açık"}`}</Text>
        )}
      </div>
      {details && (
        <pre className="overflow-auto whitespace-pre-wrap break-words rounded-md bg-background-tint-02 p-3 text-xs">
          {JSON.stringify(chunk, null, 2)}
        </pre>
      )}
      {editing && (
        <ChunkEditModal
          chunk={chunk}
          onClose={() => setEditing(false)}
          onSaved={onSaved}
        />
      )}
    </article>
  );
}
