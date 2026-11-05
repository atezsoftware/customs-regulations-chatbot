"use client";

import { useState } from "react";
import { Button, Text } from "@opal/components";
import { SvgChevronDown, SvgChevronRight } from "@opal/icons";
import type {
  LegislationSearchFileGroup,
  LegislationSearchResult,
} from "@/lib/legistations/interfaces";
import LegislationSearchChunkCard from "@/sections/cards/LegislationSearchChunkCard";
import { legislationDisplayName } from "@/lib/legistations/sourceIdentity";
import LegislationSourceIdentity from "@/sections/legistations/LegislationSourceIdentity";
import type { LegislationFile } from "@/lib/legistations/interfaces";

interface LegislationSearchFileCardProps {
  group: LegislationSearchFileGroup;
  onOpenFile: (hit: LegislationSearchResult) => void;
  onSaved: () => void;
  onSelectFile?: (file: LegislationFile) => void;
}

export default function LegislationSearchFileCard({
  group,
  onOpenFile,
  onSaved,
  onSelectFile,
}: LegislationSearchFileCardProps) {
  const [expanded, setExpanded] = useState(false);
  return (
    <article className="rounded-lg border border-border-02 p-4 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1 break-words [overflow-wrap:anywhere]">
          <Text font="main-ui-action" as="h3">
            {legislationDisplayName(group.file)}
          </Text>
          <LegislationSourceIdentity
            file={group.file}
            onSelectFile={onSelectFile}
          />
          <Text
            font="secondary-body"
            color="text-03"
          >{`${group.hits.length} chunk`}</Text>
        </div>
        <Button
          icon={expanded ? SvgChevronDown : SvgChevronRight}
          prominence="tertiary"
          aria-expanded={expanded}
          aria-label={`${group.file.name} · ${group.hits.length} chunk`}
          onClick={() => setExpanded(!expanded)}
        />
      </div>
      {expanded && (
        <div className="flex flex-col gap-3">
          {group.hits.slice(0, 3).map((hit) => (
            <LegislationSearchChunkCard
              key={hit.chunk.id}
              result={hit}
              onOpenFile={() => onOpenFile(hit)}
              onSaved={onSaved}
            />
          ))}
          {group.hits.length > 3 && (
            <Text
              font="secondary-body"
              color="text-03"
            >{`${group.hits.length - 3} eşleşme daha var.`}</Text>
          )}
          <Button
            size="sm"
            prominence="secondary"
            onClick={() => onOpenFile(group.hits[0]!)}
          >
            Daha fazla göster
          </Button>
        </div>
      )}
    </article>
  );
}
