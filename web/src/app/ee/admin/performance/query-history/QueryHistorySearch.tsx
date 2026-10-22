import { useEffect, useState } from "react";
import { Button, InputTypeIn, Text } from "@opal/components";
import type { QueryHistoryFilters } from "@/app/ee/admin/performance/query-history/filters";

interface QueryHistorySearchProps {
  filters: QueryHistoryFilters;
  allowIdentity: boolean;
  onApply: (filters: QueryHistoryFilters) => void;
}

export default function QueryHistorySearch({
  filters,
  allowIdentity,
  onApply,
}: QueryHistorySearchProps) {
  const selectedUser =
    filters.user_query || filters.user_email || filters.user_id || "";
  const [userQuery, setUserQuery] = useState(selectedUser);
  const [apiKeyQuery, setApiKeyQuery] = useState(filters.api_key_query || "");
  const [query, setQuery] = useState(filters.q || "");

  useEffect(() => setUserQuery(selectedUser), [selectedUser]);
  useEffect(
    () => setApiKeyQuery(filters.api_key_query || ""),
    [filters.api_key_query]
  );
  useEffect(() => setQuery(filters.q || ""), [filters.q]);

  return (
    <form
      className="flex flex-col gap-3 pb-4"
      onSubmit={(event) => {
        event.preventDefault();
        const next: QueryHistoryFilters = {
          ...filters,
          q: query.trim() || undefined,
        };
        if (allowIdentity) {
          next.api_key_query = apiKeyQuery.trim() || undefined;
          if (userQuery.trim() !== selectedUser) {
            delete next.user_id;
            delete next.user_email;
            next.user_query = userQuery.trim() || undefined;
          }
        }
        onApply(next);
      }}
    >
      <div className="grid gap-3 md:grid-cols-3">
        {allowIdentity && (
          <>
            <div className="flex flex-col gap-1">
              <label htmlFor="query-history-user">
                <Text font="main-ui-action">Kullanıcı e-postası / adı</Text>
              </label>
              <InputTypeIn
                id="query-history-user"
                placeholder="E-posta veya dış kullanıcı adı"
                value={userQuery}
                maxLength={255}
                onChange={(event) => setUserQuery(event.target.value)}
                searchIcon
              />
              {(filters.user_id || filters.user_email) && (
                <Text font="secondary-body" color="text-03">
                  Seçili hesabın tüm sohbetleri. Aramayı düzenleyerek başka
                  kullanıcı bulabilirsiniz.
                </Text>
              )}
            </div>
            <div className="flex flex-col gap-1">
              <label htmlFor="query-history-api-key">
                <Text font="main-ui-action">API anahtarı</Text>
              </label>
              <InputTypeIn
                id="query-history-api-key"
                placeholder="Anahtar adı, numarası veya maskeli değeri"
                value={apiKeyQuery}
                maxLength={255}
                onChange={(event) => setApiKeyQuery(event.target.value)}
              />
            </div>
          </>
        )}
        <div className="flex flex-col gap-1">
          <label htmlFor="query-history-query">
            <Text font="main-ui-action">Soru / sohbet ara</Text>
          </label>
          <InputTypeIn
            id="query-history-query"
            placeholder="Soru metni, sohbet başlığı veya session ID"
            value={query}
            maxLength={500}
            onChange={(event) => setQuery(event.target.value)}
            searchIcon
          />
        </div>
      </div>
      {!allowIdentity && (
        <Text font="secondary-body" color="text-03">
          Anonim sorgu geçmişinde kullanıcı ve API anahtarı filtreleri
          kapalıdır.
        </Text>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Button type="submit">Ara</Button>
        <Button
          type="button"
          prominence="secondary"
          onClick={() => {
            setUserQuery("");
            setApiKeyQuery("");
            setQuery("");
            onApply({});
          }}
        >
          Filtreleri temizle
        </Button>
        <Text font="secondary-body" color="text-03">
          Arama tüm sayfalardaki sohbetlere uygulanır.
        </Text>
      </div>
    </form>
  );
}
