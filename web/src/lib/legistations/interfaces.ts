import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import type { RegulatoryIndexingProgress } from "@/lib/projects/types";

export type LegistationsSearchMode = "files" | "bm25" | "semantic" | "hybrid";

export interface LegislationFile {
  id: string;
  name: string;
  folder_path: string;
  user_id: string | null;
  file_id: string;
  status: string;
  file_type: string;
  created_at: string;
  chunk_count: number;
  document_sets: { id: number; name: string }[];
  pending_review: boolean;
  regulatory_indexing_progress?: RegulatoryIndexingProgress | null;
  index_request?: {
    attempt_id: string;
    status: "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED";
    updated_at: string;
    failure_code: string | null;
  } | null;
}

export interface LegislationFolder {
  name: string;
  path: string;
  file_count: number;
}

export interface CatalogPage<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface LegislationSearchResult {
  file: LegislationFile;
  chunk: RegulatoryChunk;
  score: number | null;
}

export interface LegislationSearchPage extends CatalogPage<LegislationSearchResult> {
  has_more: boolean;
  total_is_exact: boolean;
}

export interface LegistationsQuery {
  mode: LegistationsSearchMode;
  query: string;
  folder: string | null;
  offset: number;
}
