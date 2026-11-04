"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button, InputTypeIn, Text } from "@opal/components";
import { SettingsLayouts } from "@opal/layouts";
import { SvgArrowLeft, SvgSearch } from "@opal/icons";
import type { LegislationSearchResult } from "@/lib/legistations/interfaces";
import { simpleSearchCatalog } from "@/lib/legistations/svc";
import SelectedLegislationFile from "@/sections/legistations/SelectedLegislationFile";
import LegislationSearchChunkCard from "@/sections/cards/LegislationSearchChunkCard";

export default function SimpleSearchPage() {
  const [draft, setDraft] = useState("");
  const [request, setRequest] = useState<{
    query: string;
    sequence: number;
  } | null>(null);
  const [selectedHit, setSelectedHit] =
    useState<LegislationSearchResult | null>(null);
  const search = useSWR(
    request
      ? ["legistations-simple-search", request.query, request.sequence]
      : null,
    () => simpleSearchCatalog(request!.query),
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      shouldRetryOnError: false,
    }
  );
  return (
    <SettingsLayouts.Root width="lg">
      <SettingsLayouts.Header
        icon={SvgSearch}
        title="Basit arama"
        description="Mevzuatta bulunan kaynakları ve ilgili chunkları inceleyin."
      >
        <form
          className="flex flex-wrap items-center gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (!draft.trim() || search.isLoading) return;
            setSelectedHit(null);
            setRequest((current) => ({
              query: draft.trim(),
              sequence: (current?.sequence ?? 0) + 1,
            }));
          }}
        >
          <div className="flex-1 min-w-48">
            <InputTypeIn
              aria-label="Basit mevzuat araması"
              searchIcon
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ne aramak istiyorsunuz?"
            />
          </div>
          <Button
            type="submit"
            icon={SvgSearch}
            disabled={!draft.trim() || search.isLoading}
          >
            Ara
          </Button>
        </form>
      </SettingsLayouts.Header>
      <SettingsLayouts.Body>
        {selectedHit ? (
          <div className="flex flex-col gap-4">
            <Button
              prominence="tertiary"
              icon={SvgArrowLeft}
              onClick={() => setSelectedHit(null)}
            >
              Sonuçlara dön
            </Button>
            <SelectedLegislationFile
              key={selectedHit.file.id}
              file={selectedHit.file}
              focusedChunkId={selectedHit.chunk.id}
              matchingChunks={
                search.data?.search.items
                  .filter((hit) => hit.file.id === selectedHit.file.id)
                  .map((hit) => hit.chunk) ?? []
              }
              onRenamed={(name) =>
                setSelectedHit({
                  ...selectedHit,
                  file: { ...selectedHit.file, name },
                })
              }
            />
          </div>
        ) : search.isLoading ? (
          <div role="status">
            <Text font="main-ui-body" color="text-03">
              Kaynaklar aranıyor ve bulunanlar özetleniyor…
            </Text>
          </div>
        ) : search.error ? (
          <div role="alert">
            <Text font="main-ui-body">
              {search.error instanceof Error
                ? search.error.message
                : "Arama tamamlanamadı."}
            </Text>
          </div>
        ) : search.data ? (
          <div className="flex flex-col gap-5">
            <section
              aria-label="Bulunanların özeti"
              className="flex flex-col gap-3 rounded-lg border border-border-02 bg-background-tint-01 p-4"
            >
              <Text font="heading-h3" as="h2">
                Bulduklarım
              </Text>
              {search.data.summary && (
                <div className="whitespace-pre-wrap break-words">
                  <Text font="main-ui-body" as="p">
                    {search.data.summary}
                  </Text>
                </div>
              )}
              {search.data.summary_error && (
                <div role="alert">
                  <Text font="main-ui-body">{search.data.summary_error}</Text>
                </div>
              )}
              <Text
                font="secondary-body"
                color="text-03"
              >{`${search.data.summary_model.includes("nemotron-3-ultra") ? "Nemotron 3 Ultra" : search.data.summary_model} · Bulunan kaynakların tanıtımı.`}</Text>
            </section>
            <Text
              font="main-ui-action"
              as="h2"
            >{`${search.data.search.items.length} kaynak chunk`}</Text>
            {search.data.search.items.length === 0 && (
              <Text font="main-ui-body" color="text-03">
                Eşleşen chunk bulunamadı.
              </Text>
            )}
            {search.data.search.items.map((hit, index) => (
              <LegislationSearchChunkCard
                key={hit.chunk.id}
                result={hit}
                allowEditing={false}
                sourceNumber={index + 1}
                onOpenFile={() => setSelectedHit(hit)}
              />
            ))}
          </div>
        ) : (
          <Text font="main-ui-body" color="text-03">
            Arama yapın; ilgili kaynakların özeti ve en iyi 20 chunk burada
            görünsün.
          </Text>
        )}
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
