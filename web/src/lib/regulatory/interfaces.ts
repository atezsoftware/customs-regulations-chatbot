export interface RegulatoryChunk {
  id: string;
  user_file_id: string;
  text: string;
  position: number;
  chunk_type: string | null;
  heading_path: string[];
  chunk_metadata: Record<string, unknown>;
  validity_start_date: string | null;
  validity_end_date: string | null;
  status: "active" | "superseded";
  source: "indexed" | "amendment";
  supersedes_chunk_id: string | null;
  superseded_by_chunk_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface RegulatoryChunkPage {
  items: RegulatoryChunk[];
  total: number;
  offset: number;
  limit: number;
}

export interface RegulatoryChunkUpdate {
  text?: string;
  heading_path?: string[];
  chunk_metadata?: Record<string, unknown>;
  validity_start_date?: string;
  clear_validity_start_date?: boolean;
  validity_end_date?: string;
  clear_validity_end_date?: boolean;
}
export interface AmendmentRuntime {
  analysis_model?: "gemini-3.8-flash" | "gemini-3.5-flash-lite";
  batch_id: number;
  status: string;
  stage: string;
  lease_generation: number;
  scope: "worker_container";
  raw_text_chars: number;
  current_bytes: number | null;
  limit_bytes: number | null;
  peak_bytes: number | null;
  reserve_bytes: number | null;
  active: number | null;
  peak_active: number | null;
  max_parallel: number | null;
  admission_limited: boolean | null;
  dependency_limited: boolean | null;
  calibrating: boolean | null;
  memory_checked_at: string | null;
  activity_checked_at: string | null;
}
