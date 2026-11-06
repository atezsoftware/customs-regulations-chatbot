"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button, InputTypeIn, Text } from "@opal/components";
import { SettingsLayouts } from "@opal/layouts";
import { SvgArrowLeft, SvgSearch } from "@opal/icons";
import type { LegislationSearchResult } from "@/lib/legistations/interfaces";
import { simpleSearchCatalog } from "@/lib/legistations/svc";
import SelectedLegislationFile from "@/sections/legistations/SelectedLegislationFile";
import LegislationSearchFileCard from "@/sections/cards/LegislationSearchFileCard";
import { groupLegislationSearchResults } from "@/lib/legistations/hooks";

export default function SimpleSearchPage() {
  const [draft, setDraft] = useState("");
  const [request, setRequest] = useState<{
    query: string;
    sequence: number;
  } | null>(null);
  const [selectedHit, setSelectedHit] =
    useState<LegislationSearchResult | null>(null);
  const searchKey:
    | readonly ["legistations-simple-search", string, number]
    | null = request
    ? ["legistations-simple-search", request.query, request.sequence]
    : null;
  const search = useSWR(searchKey, ([, query]) => simpleSearchCatalog(query), {
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
    shouldRetryOnError: false,
  });
  const rewrittenQuery = search.data?.query_rewritten
    ? search.data.search_query?.trim()
    : null;
  const groups = groupLegislationSearchResults(search.data?.search.items ?? []);
  const sourceNumbers = new Map(
    search.data?.search.items.map((hit, index) => [hit.chunk.id, index + 1]) ??
      []
  );
  const providerCost = search.data?.summary_usage?.provider_cost_usd;
  const summaryCostLabel =
    typeof providerCost === "number" &&
    Number.isFinite(providerCost) &&
    providerCost >= 0
      ? `Nemotron maliyeti: $${providerCost.toFixed(6)}`
      : null;
  return (
    <SettingsLayouts.Root width="lg">
      <SettingsLayouts.Header
        icon={SvgSearch}
        title="Basit arama"
        description="Mevzuatta bulunan kaynakları ve ilgili chunkları inceleyin."
      >
        <form
          className="flex w-full flex-col items-stretch gap-2 sm:flex-row sm:items-center"
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
          <div className="w-full min-w-0 sm:flex-1">
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
            {rewrittenQuery && rewrittenQuery !== request?.query && (
              <Text
                font="secondary-body"
                color="text-03"
                as="p"
              >{`Aranan ifade: ${rewrittenQuery}`}</Text>
            )}
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
              >{`${search.data.summary_model.includes("nemotron-3-ultra") ? "Nemotron 3 Ultra" : search.data.summary_model} · Seçili kaynak pasajları ve başlık yapısına dayalı kısa özet.`}</Text>
              {summaryCostLabel && (
                <Text font="secondary-body" color="text-03">
                  {summaryCostLabel}
                </Text>
              )}
            </section>
            <Text
              font="main-ui-action"
              as="h2"
            >{`${groups.length} mevzuat · ${search.data.search.items.length} chunk`}</Text>
            {search.data.search.scope?.heading_prefixes.map((prefix) => (
              <Text
                key={prefix.join(" > ")}
                font="secondary-body"
                color="text-03"
              >{`Arama kapsamı: ${prefix.join(" > ")}`}</Text>
            ))}
            {search.data.search.items.length === 0 && (
              <Text font="main-ui-body" color="text-03">
                Eşleşen chunk bulunamadı.
              </Text>
            )}
            {groups.map((group) => (
              <LegislationSearchFileCard
                key={`${request?.sequence}-${group.file.id}`}
                group={group}
                allowEditing={false}
                showAllMatches
                sourceNumbers={sourceNumbers}
                onOpenFile={setSelectedHit}
              />
            ))}
          </div>
        ) : (
          <Text font="main-ui-body" color="text-03">
            Arama yapın; ilgili kaynakların özeti, en az 20 uygun chunk ve
            normalize skoru 0,8 ve üzerindeki diğer eşleşmeler burada görünsün.
          </Text>
        )}
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
