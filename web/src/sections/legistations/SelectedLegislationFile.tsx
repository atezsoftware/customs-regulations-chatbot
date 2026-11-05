"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button, Text } from "@opal/components";
import type { LegislationFile } from "@/lib/legistations/interfaces";
import {
  catalogFileIsIndexing,
  useCatalogFile,
} from "@/lib/legistations/hooks";
import {
  catalogChunkLocationUrl,
  fetchCatalogChunkLocation,
  indexCatalogFile,
} from "@/lib/legistations/svc";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import RegulatoryFileDetails from "@/sections/document-sets/RegulatoryFileDetails";
import LegislationSourceIdentity from "@/sections/legistations/LegislationSourceIdentity";

interface SelectedLegislationFileProps {
  file: LegislationFile;
  onRenamed: (name: string) => void;
  onSelectFile?: (file: LegislationFile) => void;
  matchingChunks?: RegulatoryChunk[];
  focusedChunkId?: string;
}

export default function SelectedLegislationFile({
  file,
  onRenamed,
  onSelectFile,
  matchingChunks = [],
  focusedChunkId,
}: SelectedLegislationFileProps) {
  const { data, mutate, error: refreshError } = useCatalogFile(file.id);
  const [focusedMatchId, setFocusedMatchId] = useState(
    focusedChunkId ?? matchingChunks[0]?.id
  );
  const location = useSWR(
    focusedMatchId ? catalogChunkLocationUrl(file.id, focusedMatchId) : null,
    () => fetchCatalogChunkLocation(file.id, focusedMatchId!),
    { revalidateOnFocus: false }
  );
  const currentFile = data ?? file;
  const progress = currentFile.regulatory_indexing_progress;
  const active = catalogFileIsIndexing(currentFile);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chunked = currentFile.status === "CHUNKED";

  async function indexFile() {
    setIndexing(true);
    setError(null);
    try {
      const updated = await indexCatalogFile(file.id);
      await mutate(updated, false);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "İndeksleme başlatılamadı."
      );
    } finally {
      setIndexing(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <LegislationSourceIdentity
        file={currentFile}
        onSelectFile={onSelectFile}
      />
      {currentFile.pending_review && (
        <Text font="main-ui-body">
          Bu dosya inceleme bekliyor. Düzenlemeleri Updates → New document
          ekranından yapın.
        </Text>
      )}
      {chunked && !currentFile.pending_review && (
        <div className="flex flex-col gap-2 rounded-lg border border-border-02 p-3">
          <Text font="main-ui-body">
            Bu dosyanın chunkları henüz arama indeksine yayımlanmadı.
            Düzenlemeler kaydedilir; aramaya açmak için İndeksle düğmesini
            kullanın.
          </Text>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              disabled={indexing || active}
              onClick={() => void indexFile()}
            >
              {indexing
                ? "Başlatılıyor…"
                : active
                  ? "İndeksleniyor…"
                  : "İndeksle"}
            </Button>
          </div>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Text
          font="secondary-body"
          color="text-03"
        >{`Dosya durumu: ${currentFile.status}`}</Text>
        <Button size="sm" prominence="tertiary" onClick={() => void mutate()}>
          Durumu yenile
        </Button>
      </div>
      {progress && (
        <Text
          font="secondary-body"
          color="text-03"
        >{`İndeksleme: ${progress.status} · ${progress.stage} · ${progress.completed_items}/${progress.total_items}${progress.error_summary ? ` · ${progress.error_summary}` : ""}`}</Text>
      )}
      {!progress && currentFile.index_request && (
        <Text
          font="secondary-body"
          color="text-03"
        >{`İndeksleme isteği: ${currentFile.index_request.status}${currentFile.index_request.failure_code ? ` · ${currentFile.index_request.failure_code}` : ""}`}</Text>
      )}
      {refreshError && (
        <Text font="secondary-body" color="text-03">
          Dosyanın güncel durumu alınamadı.
        </Text>
      )}
      {error && (
        <div role="alert">
          <Text font="main-ui-body">{error}</Text>
        </div>
      )}
      {matchingChunks.length > 0 && (
        <section
          aria-label="Arama eşleşmeleri"
          className="flex flex-col gap-2 rounded-lg border border-theme-yellow-05 bg-theme-yellow-01 p-3"
        >
          <Text font="main-ui-action">{`${matchingChunks.length} arama eşleşmesi sarıyla vurgulanır.`}</Text>
          <div className="flex flex-wrap gap-2">
            {matchingChunks.map((chunk) => (
              <Button
                key={chunk.id}
                size="sm"
                prominence={
                  focusedMatchId === chunk.id ? "primary" : "secondary"
                }
                onClick={() => setFocusedMatchId(chunk.id)}
              >{`Chunk #${chunk.position}`}</Button>
            ))}
          </div>
        </section>
      )}
      {location.error ? (
        <div role="alert" className="flex flex-col gap-2">
          <Text font="main-ui-body">
            Arama eşleşmesinin dosyadaki konumu alınamadı.
          </Text>
          <Button
            size="sm"
            prominence="secondary"
            onClick={() => void location.mutate()}
          >
            Tekrar dene
          </Button>
        </div>
      ) : focusedMatchId && !location.data ? (
        <Text font="main-ui-body" color="text-03">
          Arama eşleşmesi açılıyor…
        </Text>
      ) : (
        <RegulatoryFileDetails
          key={`${currentFile.id}:${currentFile.status}:${focusedMatchId ?? ""}:${location.data?.offset ?? 0}`}
          initialChunkOffset={location.data?.offset ?? 0}
          highlightedChunkIds={matchingChunks.map((chunk) => chunk.id)}
          focusedChunkId={focusedMatchId}
          file={currentFile}
          readOnly={currentFile.pending_review || indexing || active}
          indexed={!chunked}
          onFileRenamed={(name) => {
            void mutate({ ...currentFile, name }, false);
            onRenamed(name);
          }}
        />
      )}
    </div>
  );
}
