"use client";

import { useState } from "react";
import { Button, InputTypeIn, Text } from "@opal/components";
import { SettingsLayouts } from "@opal/layouts";
import { SvgArrowLeft, SvgBookOpen, SvgFolder, SvgSearch } from "@opal/icons";
import { cn } from "@opal/utils";
import {
  groupLegislationSearchResults,
  useCatalogFiles,
  useCatalogSearch,
} from "@/lib/legistations/hooks";
import type {
  LegislationFile,
  LegistationsQuery,
  LegistationsSearchMode,
  LegislationSearchResult,
} from "@/lib/legistations/interfaces";
import { CATALOG_PAGE_SIZE } from "@/lib/legistations/svc";
import FolderTree from "@/sections/legistations/FolderTree";
import SelectedLegislationFile from "@/sections/legistations/SelectedLegislationFile";
import LegislationSearchFileCard from "@/sections/cards/LegislationSearchFileCard";
import LegislationSourceIdentity from "@/sections/legistations/LegislationSourceIdentity";
import { legislationDisplayName } from "@/lib/legistations/sourceIdentity";

const SEARCH_MODES: { mode: LegistationsSearchMode; label: string }[] = [
  { mode: "files", label: "Dosya adı" },
  { mode: "hybrid", label: "İçerik araması" },
];

interface CatalogPaginationProps {
  offset: number;
  count: number;
  total: number;
  hasMore: boolean;
  onChange: (offset: number) => void;
}

function CatalogPagination({
  offset,
  count,
  total,
  hasMore,
  onChange,
}: CatalogPaginationProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <Button
        prominence="secondary"
        size="sm"
        disabled={offset === 0}
        onClick={() => onChange(Math.max(0, offset - CATALOG_PAGE_SIZE))}
      >
        Önceki
      </Button>
      <Text
        font="secondary-body"
        color="text-03"
      >{`${total === 0 ? 0 : offset + 1}–${offset + count} / ${total}`}</Text>
      <Button
        prominence="secondary"
        size="sm"
        disabled={!hasMore}
        onClick={() => onChange(offset + CATALOG_PAGE_SIZE)}
      >
        Sonraki
      </Button>
    </div>
  );
}

export default function LegistationsPage() {
  const [mode, setMode] = useState<LegistationsSearchMode>("files");
  const [draft, setDraft] = useState("");
  const [state, setState] = useState<LegistationsQuery>({
    mode: "files",
    query: "",
    folder: null,
    offset: 0,
  });
  const [selectedFile, setSelectedFile] = useState<LegislationFile | null>(
    null
  );
  const [focusedHit, setFocusedHit] = useState<LegislationSearchResult | null>(
    null
  );
  const [foldersVisible, setFoldersVisible] = useState(true);
  const files = useCatalogFiles(state, selectedFile === null);
  const search = useCatalogSearch(state, true);
  const response = state.mode === "files" ? files : search;

  function openRelatedFile(file: LegislationFile) {
    setFocusedHit(null);
    setSelectedFile(file);
  }

  function openSearchFile(hit: LegislationSearchResult) {
    setFocusedHit(hit);
    setSelectedFile(hit.file);
  }

  function selectFolder(folder: string | null) {
    setSelectedFile(null);
    setFocusedHit(null);
    setMode("files");
    setState((current) => ({
      mode: "files",
      query: current.mode === "files" ? current.query : "",
      folder,
      offset: 0,
    }));
    if (state.mode !== "files") setDraft("");
  }

  function changeOffset(offset: number) {
    setState((current) => ({ ...current, offset }));
  }

  return (
    <SettingsLayouts.Root width="full">
      <SettingsLayouts.Header
        icon={SvgBookOpen}
        title="Legistations"
        description="Dosyaları, klasörleri ve chunkları inceleyin; düzenlenen chunkı yeniden indeksleyin."
      >
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            if (mode !== "files" && !draft.trim()) return;
            setSelectedFile(null);
            setFocusedHit(null);
            setState({
              mode,
              query: draft.trim(),
              folder: mode === "files" ? state.folder : null,
              offset: 0,
            });
          }}
        >
          <div
            className="flex flex-wrap gap-2"
            role="group"
            aria-label="Arama türü"
          >
            {SEARCH_MODES.map((item) => (
              <Button
                key={item.mode}
                size="sm"
                prominence={mode === item.mode ? "primary" : "secondary"}
                aria-pressed={mode === item.mode}
                onClick={() => setMode(item.mode)}
              >
                {item.label}
              </Button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex-1 min-w-48">
              <InputTypeIn
                aria-label="Mevzuat araması"
                searchIcon
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder={
                  mode === "files"
                    ? "Dosya adına göre ara…"
                    : "Chunk içeriğinde ara…"
                }
              />
            </div>
            <Button
              type="submit"
              icon={SvgSearch}
              disabled={mode !== "files" && !draft.trim()}
            >
              Ara
            </Button>
            <Button
              prominence="secondary"
              icon={SvgFolder}
              aria-expanded={foldersVisible}
              onClick={() => setFoldersVisible(!foldersVisible)}
            >
              {foldersVisible ? "Klasörleri gizle" : "Klasörleri göster"}
            </Button>
          </div>
          <Text font="secondary-body" color="text-03">
            İçerik araması en fazla 100 adayı yeniden sıralar; dosya başına en
            fazla 8 chunk ile 20 sonuç gösterir.
          </Text>
        </form>
      </SettingsLayouts.Header>
      <SettingsLayouts.Body>
        <div
          className={cn(
            "grid items-start gap-5 min-w-0",
            foldersVisible && "lg:grid-cols-[17rem_minmax(0,1fr)]"
          )}
        >
          {foldersVisible && (
            <aside className="rounded-lg border border-border-02 p-3 max-h-[60vh] overflow-auto">
              <FolderTree
                selected={state.folder}
                query={state.mode === "files" ? state.query : ""}
                onSelect={selectFolder}
              />
            </aside>
          )}
          <section
            className="min-w-0 flex flex-col gap-4"
            aria-label="Mevzuat dosyaları ve chunklar"
          >
            {selectedFile ? (
              <>
                <Button
                  icon={SvgArrowLeft}
                  prominence="tertiary"
                  size="sm"
                  onClick={() => setSelectedFile(null)}
                >
                  Sonuçlara dön
                </Button>
                <SelectedLegislationFile
                  key={selectedFile.id}
                  file={selectedFile}
                  onSelectFile={openRelatedFile}
                  matchingChunks={
                    search.data?.items
                      .filter((hit) => hit.file.id === selectedFile.id)
                      .map((hit) => hit.chunk) ?? []
                  }
                  focusedChunkId={focusedHit?.chunk.id}
                  onRenamed={(name) => {
                    setSelectedFile({ ...selectedFile, name });
                    void files.mutate();
                  }}
                />
              </>
            ) : (
              <>
                <Text font="main-ui-action" as="h2">
                  {state.mode === "files"
                    ? state.folder === null
                      ? "Tüm dosyalar"
                      : state.folder || "Kökteki dosyalar"
                    : `Chunk sonuçları · ${state.query}`}
                </Text>
                {response.isLoading ? (
                  <Text font="main-ui-body" color="text-03">
                    Yükleniyor…
                  </Text>
                ) : response.error ? (
                  <div role="alert" className="flex flex-col gap-2">
                    <Text font="main-ui-body">
                      {response.error instanceof Error
                        ? response.error.message
                        : "Veriler yüklenemedi."}
                    </Text>
                    <Button
                      size="sm"
                      prominence="secondary"
                      onClick={() => void response.mutate()}
                    >
                      Tekrar dene
                    </Button>
                  </div>
                ) : state.mode === "files" ? (
                  <>
                    {files.data?.items.length === 0 && (
                      <Text font="main-ui-body" color="text-03">
                        Dosya bulunamadı.
                      </Text>
                    )}
                    {files.data?.items.map((file) => (
                      <article
                        key={file.id}
                        className="flex flex-wrap justify-between items-start gap-3 rounded-lg border border-border-02 p-4"
                      >
                        <div className="flex-1 min-w-0 flex flex-col gap-1 break-words [overflow-wrap:anywhere]">
                          <Text font="main-ui-action" as="h3">
                            {legislationDisplayName(file)}
                          </Text>
                          <LegislationSourceIdentity
                            file={file}
                            onSelectFile={openRelatedFile}
                          />
                          <Text
                            font="secondary-body"
                            color="text-03"
                          >{`${file.chunk_count} chunk · ${file.status}${file.document_sets.length ? ` · ${file.document_sets.map((set) => set.name).join(", ")}` : ""}`}</Text>
                          {file.pending_review && (
                            <Text font="secondary-body" color="text-03">
                              İnceleme bekliyor
                            </Text>
                          )}
                        </div>
                        <Button
                          size="sm"
                          prominence="secondary"
                          onClick={() => {
                            setFocusedHit(null);
                            setSelectedFile(file);
                          }}
                        >
                          Dosyayı aç
                        </Button>
                      </article>
                    ))}
                    {files.data && (
                      <CatalogPagination
                        offset={state.offset}
                        count={files.data.items.length}
                        total={files.data.total}
                        hasMore={
                          state.offset + CATALOG_PAGE_SIZE < files.data.total
                        }
                        onChange={changeOffset}
                      />
                    )}
                  </>
                ) : (
                  <>
                    {search.data?.items.length === 0 && (
                      <Text font="main-ui-body" color="text-03">
                        Chunk bulunamadı.
                      </Text>
                    )}
                    {search.data && !search.data.total_is_exact && (
                      <Text font="secondary-body" color="text-03">
                        Sonuçlar aramanın en iyi eşleşmeleriyle sınırlıdır;
                        gösterilen toplam katalogdaki tüm eşleşmelerin sayısı
                        değildir.
                      </Text>
                    )}
                    {groupLegislationSearchResults(
                      search.data?.items ?? []
                    ).map((group) => (
                      <LegislationSearchFileCard
                        key={group.file.id}
                        group={group}
                        onOpenFile={openSearchFile}
                        onSelectFile={openRelatedFile}
                        onSaved={() => void search.mutate()}
                      />
                    ))}
                  </>
                )}
              </>
            )}
          </section>
        </div>
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
