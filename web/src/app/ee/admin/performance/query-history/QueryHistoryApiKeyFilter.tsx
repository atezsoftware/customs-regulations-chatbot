import useSWR from "swr";
import { Button, InputSelect, Text } from "@opal/components";
import { errorHandlingFetcher } from "@/lib/fetcher";
import type { QueryHistoryFilters } from "@/app/ee/admin/performance/query-history/filters";
import type { QueryHistoryApiKeyOption } from "@/app/ee/admin/performance/query-history/interfaces";

interface QueryHistoryApiKeyFilterProps {
  filters: QueryHistoryFilters;
  onApply: (filters: QueryHistoryFilters) => void;
}

export default function QueryHistoryApiKeyFilter({
  filters,
  onApply,
}: QueryHistoryApiKeyFilterProps) {
  const {
    data: options = [],
    isLoading,
    error,
    mutate,
  } = useSWR<QueryHistoryApiKeyOption[]>(
    "/api/admin/query-history/api-keys",
    errorHandlingFetcher
  );
  const value =
    filters.api_key_id?.toString() ??
    (filters.api_key_query ? "legacy" : "all");
  const selectedMissing =
    filters.api_key_id !== undefined &&
    !options.some((option) => option.api_key_id === filters.api_key_id);
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <Text font="main-ui-action">API / İstek kaynağı</Text>
      <InputSelect
        value={value}
        disabled={isLoading}
        onValueChange={(selected) => {
          if (selected === "legacy") return;
          const next = { ...filters };
          delete next.api_key_query;
          delete next.api_key_id;
          if (selected !== "all") next.api_key_id = Number(selected);
          onApply(next);
        }}
      >
        <InputSelect.Trigger
          aria-label="API / İstek kaynağı"
          placeholder="Tüm API’ler"
        />
        <InputSelect.Content>
          <InputSelect.Item value="all">Tüm API’ler</InputSelect.Item>
          {filters.api_key_query && !filters.api_key_id && (
            <InputSelect.Item value="legacy">{`Ad araması: ${filters.api_key_query}`}</InputSelect.Item>
          )}
          {selectedMissing && (
            <InputSelect.Item
              value={String(filters.api_key_id)}
            >{`API #${filters.api_key_id}`}</InputSelect.Item>
          )}
          {options.map((option) => (
            <InputSelect.Item
              key={option.api_key_id}
              value={String(option.api_key_id)}
            >
              {`${option.api_key_name} · #${option.api_key_id}`}
            </InputSelect.Item>
          ))}
        </InputSelect.Content>
      </InputSelect>
      <Text font="secondary-body" color="text-03">
        Seçim sohbet listesine ve CSV raporuna hemen uygulanır.
      </Text>
      {error && (
        <>
          <Text font="secondary-body" color="text-03">
            API listesi yüklenemedi.
          </Text>
          <Button
            type="button"
            prominence="secondary"
            onClick={() => void mutate()}
          >
            Yeniden dene
          </Button>
        </>
      )}
    </div>
  );
}
