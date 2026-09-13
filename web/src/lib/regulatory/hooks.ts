"use client";

import useSWR from "swr";

import { fetchChunkPageForFile } from "@/lib/regulatory/svc";
import type { RegulatoryChunkPage } from "@/lib/regulatory/interfaces";

export function useFileChunks(
  userFileId: string | null,
  offset: number,
  limit: number
) {
  const { data, error, isLoading, mutate } = useSWR<RegulatoryChunkPage>(
    userFileId
      ? `/api/regulatory/files/${encodeURIComponent(userFileId)}/chunks/page?offset=${offset}&limit=${limit}`
      : null,
    () => fetchChunkPageForFile(userFileId as string, offset, limit)
  );
  return {
    chunks: data?.items ?? [],
    total: data?.total ?? 0,
    error,
    isLoading,
    refreshChunks: mutate,
  };
}
