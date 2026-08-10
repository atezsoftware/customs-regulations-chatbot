"use client";

import { useMemo } from "react";

import {
  Button,
  Card,
  InputSelect,
  InputTypeIn,
  SelectCard,
  Text,
} from "@opal/components";

import type {
  BenchmarkCitationOption,
  BenchmarkExpectedCitationInput,
} from "@/lib/regulatory/benchmark";

interface BenchmarkCitationPickerProps {
  documentSetId: string;
  citations: BenchmarkExpectedCitationInput[];
  options: BenchmarkCitationOption[];
  search: string;
  onSearchChange: (value: string) => void;
  onChange: (citations: BenchmarkExpectedCitationInput[]) => void;
}

export default function BenchmarkCitationPicker({
  documentSetId,
  citations,
  options,
  search,
  onSearchChange,
  onChange,
}: BenchmarkCitationPickerProps) {
  const optionMap = useMemo(
    () => new Map(options.map((option) => [option.chunk_id, option])),
    [options]
  );
  const availableOptions = useMemo(() => {
    const selected = new Set(citations.map((citation) => citation.chunk_id));
    const query = search.trim().toLocaleLowerCase("tr");
    return options
      .filter((option) => !selected.has(option.chunk_id))
      .filter(
        (option) =>
          !query ||
          `${option.file_name} ${option.heading_path.join(" ")} ${option.text_excerpt}`
            .toLocaleLowerCase("tr")
            .includes(query)
      )
      .slice(0, 30);
  }, [citations, options, search]);

  const updateCitation = (
    chunkId: string,
    patch: Partial<BenchmarkExpectedCitationInput>
  ) =>
    onChange(
      citations.map((citation) =>
        citation.chunk_id === chunkId ? { ...citation, ...patch } : citation
      )
    );

  return (
    <Card border="solid" padding="md">
      <div className="mb-3">
        <Text font="main-ui-action" color="text-04">
          Expected citations
        </Text>
        <Text as="p" font="secondary-body" color="text-02">
          Select exact chunks from the document set. Required citations affect
          deterministic recall and judge scoring.
        </Text>
      </div>
      {!documentSetId ? (
        <Text font="main-ui-muted" color="text-02">
          Select a document set to browse its chunks.
        </Text>
      ) : (
        <>
          <InputTypeIn
            aria-label="Search expected citations"
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="Search file, article, heading, or chunk text"
          />
          {search && availableOptions.length > 0 && (
            <div className="mt-2 max-h-56 overflow-y-auto rounded-lg border border-border-02 bg-background">
              {availableOptions.map((option) => (
                <SelectCard
                  key={option.chunk_id}
                  state="filled"
                  role="button"
                  tabIndex={0}
                  padding="sm"
                  rounding="xs"
                  aria-label={`Add citation ${option.file_name}`}
                  onClick={() => {
                    onChange([
                      ...citations,
                      {
                        chunk_id: option.chunk_id,
                        requirement: "required",
                        notes: null,
                      },
                    ]);
                    onSearchChange("");
                  }}
                  onKeyDown={(event) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    event.currentTarget.click();
                  }}
                >
                  <div className="flex flex-col gap-1 text-left">
                    <Text font="main-ui-action" color="text-04">
                      {`${option.file_name} · ${option.heading_path.join(" › ")}`}
                    </Text>
                    <Text font="secondary-body" color="text-02" maxLines={2}>
                      {option.text_excerpt}
                    </Text>
                  </div>
                </SelectCard>
              ))}
            </div>
          )}
        </>
      )}
      <div className="mt-3 flex flex-col gap-2">
        {citations.map((citation) => {
          const option = optionMap.get(citation.chunk_id);
          return (
            <Card key={citation.chunk_id} border="solid" padding="sm">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <Text font="main-ui-action" color="text-04">
                    {option?.file_name ?? citation.chunk_id}
                  </Text>
                  <Text font="secondary-body" color="text-02" maxLines={1}>
                    {option?.heading_path.join(" › ") ?? "Saved citation"}
                  </Text>
                </div>
                <Button
                  size="sm"
                  variant="danger"
                  prominence="tertiary"
                  onClick={() =>
                    onChange(
                      citations.filter(
                        (item) => item.chunk_id !== citation.chunk_id
                      )
                    )
                  }
                >
                  Remove
                </Button>
              </div>
              <div className="mt-2 grid gap-2 md:grid-cols-[150px_1fr]">
                <InputSelect
                  value={citation.requirement}
                  onValueChange={(value) =>
                    updateCitation(citation.chunk_id, {
                      requirement: value as "required" | "supporting",
                    })
                  }
                >
                  <InputSelect.Trigger aria-label="Citation requirement" />
                  <InputSelect.Content>
                    <InputSelect.Item value="required">
                      Required
                    </InputSelect.Item>
                    <InputSelect.Item value="supporting">
                      Supporting
                    </InputSelect.Item>
                  </InputSelect.Content>
                </InputSelect>
                <InputTypeIn
                  aria-label={`Notes for ${option?.file_name ?? citation.chunk_id}`}
                  value={citation.notes ?? ""}
                  onChange={(event) =>
                    updateCitation(citation.chunk_id, {
                      notes: event.target.value || null,
                    })
                  }
                  placeholder="Why this citation is expected (optional)"
                />
              </div>
            </Card>
          );
        })}
      </div>
    </Card>
  );
}
