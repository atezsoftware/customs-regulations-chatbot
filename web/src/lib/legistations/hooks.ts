"use client";

import useSWR from "swr";
import type {
  LegislationFile,
  LegistationsQuery,
  LegislationSearchResult,
  LegislationSearchFileGroup,
} from "@/lib/legistations/interfaces";
import {
  catalogFilesUrl,
  catalogFileUrl,
  catalogFoldersUrl,
  fetchCatalogFiles,
  fetchCatalogFile,
  fetchCatalogFolders,
  searchCatalogChunks,
} from "@/lib/legistations/svc";

export function catalogFileIsIndexing(file: LegislationFile): boolean {
  return (
    file.status === "INDEXING" ||
    file.index_request?.status === "PENDING" ||
    file.index_request?.status === "RUNNING" ||
    (file.regulatory_indexing_progress !== null &&
      file.regulatory_indexing_progress !== undefined &&
      ["QUEUED", "RUNNING", "RETRY_WAIT", "CANCELLING"].includes(
        file.regulatory_indexing_progress.status
      ))
  );
}

export function useCatalogFile(fileId: string) {
  return useSWR(catalogFileUrl(fileId), () => fetchCatalogFile(fileId), {
    refreshInterval: (file) => (file && catalogFileIsIndexing(file) ? 3000 : 0),
  });
}

export function useCatalogFiles(state: LegistationsQuery, enabled: boolean) {
  return useSWR(
    enabled && state.mode === "files" ? catalogFilesUrl(state) : null,
    () => fetchCatalogFiles(state)
  );
}

export function useCatalogSearch(state: LegistationsQuery, enabled: boolean) {
  return useSWR(
    enabled && state.mode !== "files" && state.query.trim()
      ? ["legistations-search", state.mode, state.query, state.offset]
      : null,
    () => searchCatalogChunks(state),
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
      shouldRetryOnError: false,
    }
  );
}

export function useCatalogFolders(
  parent: string,
  query: string,
  offset: number
) {
  return useSWR(catalogFoldersUrl(parent, query, offset), () =>
    fetchCatalogFolders(parent, query, offset)
  );
}

export function groupLegislationSearchResults(
  hits: LegislationSearchResult[]
): LegislationSearchFileGroup[] {
  const groups = new Map<string, LegislationSearchFileGroup>();
  for (const hit of hits) {
    const group = groups.get(hit.file.id);
    if (group) group.hits.push(hit);
    else groups.set(hit.file.id, { file: hit.file, hits: [hit] });
  }
  return Array.from(groups.values());
}
