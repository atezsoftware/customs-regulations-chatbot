import QueryHistoryApiKeyFilter from "@/app/ee/admin/performance/query-history/QueryHistoryApiKeyFilter";
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
  const [query, setQuery] = useState(filters.q || "");

  useEffect(() => setUserQuery(selectedUser), [selectedUser]);
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
          if (userQuery.trim() !== selectedUser) {
            delete next.user_id;
            delete next.user_email;
            next.user_query = userQuery.trim() || undefined;
          }
        }
        onApply(next);
      }}
    >
      <Text font="heading-h3">Sohbet ve rapor filtreleri</Text>
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
            <QueryHistoryApiKeyFilter filters={filters} onApply={onApply} />
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
