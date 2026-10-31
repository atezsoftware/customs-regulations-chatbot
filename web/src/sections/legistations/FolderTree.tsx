"use client";

import { useState } from "react";
import { Button, Text } from "@opal/components";
import { SvgChevronDown, SvgChevronRight, SvgFolder } from "@opal/icons";
import { cn } from "@opal/utils";
import { useCatalogFolders } from "@/lib/legistations/hooks";
import { FOLDER_PAGE_SIZE } from "@/lib/legistations/svc";
import type { LegislationFolder } from "@/lib/legistations/interfaces";

interface FolderTreeProps {
  selected: string | null;
  query: string;
  onSelect: (path: string | null) => void;
}

interface FolderChildrenProps extends FolderTreeProps {
  parent: string;
}

interface FolderNodeProps extends FolderTreeProps {
  folder: LegislationFolder;
}

function FolderNode({ folder, selected, query, onSelect }: FolderNodeProps) {
  const [expanded, setExpanded] = useState(false);
  return (
    <li className="min-w-0">
      <div
        className={cn(
          "flex items-center gap-1 rounded-md",
          selected === folder.path && "bg-action-selection-01"
        )}
      >
        <Button
          icon={expanded ? SvgChevronDown : SvgChevronRight}
          size="sm"
          prominence="tertiary"
          aria-label={`${folder.name} klasörünü ${expanded ? "daralt" : "genişlet"}`}
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        />
        <div className="min-w-0 overflow-hidden">
          <Button
            icon={SvgFolder}
            size="sm"
            prominence="tertiary"
            tooltip={folder.path}
            aria-pressed={selected === folder.path}
            onClick={() => {
              setExpanded(true);
              onSelect(folder.path);
            }}
          >
            {`${folder.name} (${folder.file_count})`}
          </Button>
        </div>
      </div>
      {expanded && (
        <div className="pl-4 border-l border-border-01">
          <FolderChildren
            parent={folder.path}
            selected={selected}
            query={query}
            onSelect={onSelect}
          />
        </div>
      )}
    </li>
  );
}

function FolderChildren({
  parent,
  selected,
  query,
  onSelect,
}: FolderChildrenProps) {
  const [offset, setOffset] = useState(0);
  const { data, isLoading, error } = useCatalogFolders(parent, query, offset);
  return (
    <div className="flex flex-col gap-1">
      {isLoading && (
        <Text font="secondary-body" color="text-03">
          Klasörler yükleniyor…
        </Text>
      )}
      {error && <Text font="secondary-body">Klasörler yüklenemedi.</Text>}
      <ul className="flex flex-col gap-1">
        {data?.items.map((folder) => (
          <FolderNode
            key={folder.path}
            folder={folder}
            selected={selected}
            query={query}
            onSelect={onSelect}
          />
        ))}
      </ul>
      {data && data.total > FOLDER_PAGE_SIZE && (
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            prominence="tertiary"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - FOLDER_PAGE_SIZE))}
          >
            Önceki klasörler
          </Button>
          <Button
            size="sm"
            prominence="tertiary"
            disabled={offset + FOLDER_PAGE_SIZE >= data.total}
            onClick={() => setOffset(offset + FOLDER_PAGE_SIZE)}
          >
            Sonraki klasörler
          </Button>
        </div>
      )}
    </div>
  );
}

export default function FolderTree({
  selected,
  query,
  onSelect,
}: FolderTreeProps) {
  return (
    <nav aria-label="Mevzuat klasörleri" className="flex flex-col gap-3">
      <Text font="main-ui-action">Klasörler</Text>
      <Button
        prominence="tertiary"
        size="sm"
        aria-pressed={selected === null}
        onClick={() => onSelect(null)}
      >
        Tüm dosyalar
      </Button>
      <Button
        prominence="tertiary"
        size="sm"
        aria-pressed={selected === ""}
        onClick={() => onSelect("")}
      >
        Kökteki dosyalar
      </Button>
      <FolderChildren
        key={query}
        parent=""
        selected={selected}
        query={query}
        onSelect={onSelect}
      />
    </nav>
  );
}
