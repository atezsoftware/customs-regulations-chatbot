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

export interface RegulatoryChunkUpdate {
  text?: string;
  heading_path?: string[];
  chunk_metadata?: Record<string, unknown>;
  validity_start_date?: string;
  clear_validity_start_date?: boolean;
  validity_end_date?: string;
  clear_validity_end_date?: boolean;
}
