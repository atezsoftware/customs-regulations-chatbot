export interface LabelingTaxonomySummary {
  id: string;
  name: string;
  version_hash: string;
  label_count: number;
  created_at: string;
}

export interface LabelingProvider {
  id: number;
  name: string;
}

export interface LabelingSetup {
  model: string;
  taxonomies: LabelingTaxonomySummary[];
  providers: LabelingProvider[];
  counts: {
    files: number;
    canonical_chunks: number;
    derived_chunks: number;
  };
  active_run_id: string | null;
  warnings: string[];
}

export interface TaxonomyLabelInput {
  id: string;
  name: string;
  description: string;
}

export interface TaxonomyInput {
  name: string;
  labels: TaxonomyLabelInput[];
}

export type LabelingRunStatus =
  | "queued"
  | "running"
  | "completed"
  | "completed_with_errors"
  | "failed"
  | "cancelled";

export type LabelingRunStage =
  | "preparing"
  | "submitting"
  | "waiting"
  | "applying"
  | "projecting"
  | "finished";

export interface LabelingRun {
  id: string;
  document_set_id: number;
  taxonomy_id: string;
  taxonomy_name: string;
  model: string;
  status: LabelingRunStatus;
  stage: LabelingRunStage;
  total_chunks: number;
  completed_chunks: number;
  failed_chunks: number;
  stale_chunks: number;
  derived_chunks: number;
  unresolved_derived_chunks: number;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
  error: string | null;
  cancel_requested: boolean;
}

export interface LabelingRunItem {
  chunk_id: string;
  file_id: string;
  status: string;
  labels: string[];
  error: string | null;
}

export interface LabelingRunItemsPage {
  items: LabelingRunItem[];
  total: number;
}
