import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";
import type { RegulatoryIndexingProgress } from "@/lib/projects/types";

export type LegistationsSearchMode = "files" | "hybrid";

export interface IntakeOriginalSource {
  asset_id: string;
  sha256: string;
  original_name: string;
  original_url: string | null;
  final_url: string | null;
}

export interface IntakeRelatedFile {
  user_file_id: string;
  document_id: string;
  title: string;
  source: IntakeOriginalSource;
}

export interface IntakeFileSourceLineage {
  version: 1;
  intake_id: string;
  package_id: string;
  document_set_id: number;
  document_id: string;
  role: "root" | "annex";
  source: IntakeOriginalSource;
  parents: IntakeRelatedFile[];
  attachments: IntakeRelatedFile[];
}

export interface ReviewedRelatedFile {
  user_file_id: string;
  title: string;
  original_name: string | null;
  source_url: string | null;
  document_set_id: number;
  intake_id: string;
  review_decision_id: string;
  basis: "reviewed_existing_parent";
}

export interface ReviewedFileRelations {
  parents: ReviewedRelatedFile[];
  attachments: ReviewedRelatedFile[];
}

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
  source_lineage?: IntakeFileSourceLineage | null;
  reviewed_relations?: ReviewedFileRelations | null;
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
  candidate_limit: number;
  result_limit: number;
  per_file_limit: number;
}

export interface LegistationsQuery {
  mode: LegistationsSearchMode;
  query: string;
  folder: string | null;
  offset: number;
}

export interface LegislationSummaryUsage {
  response_id: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  provider_cost_usd: number | null;
}

export interface LegislationSimpleSearchResponse {
  search: LegislationSearchPage;
  summary: string | null;
  summary_model: string;
  summary_error: string | null;
  search_query?: string;
  query_rewritten?: boolean;
  summary_usage?: LegislationSummaryUsage | null;
  summary_error_code?: string | null;
  request_id?: string;
}

export interface LegislationSearchFileGroup {
  file: LegislationFile;
  hits: LegislationSearchResult[];
}
