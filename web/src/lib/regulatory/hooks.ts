"use client";

import useSWR from "swr";

import { fetchChunksForFile } from "@/lib/regulatory/svc";
import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";

export function useFileChunks(userFileId: string | null) {
  const { data, error, isLoading, mutate } = useSWR<RegulatoryChunk[]>(
    userFileId ? `/api/regulatory/files/${userFileId}/chunks` : null,
    () => fetchChunksForFile(userFileId as string)
  );
  return {
    chunks: data ?? [],
    error,
    isLoading,
    refreshChunks: mutate,
  };
}
