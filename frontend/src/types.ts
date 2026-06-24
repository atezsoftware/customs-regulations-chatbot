export interface SafeUser {
  id: number;
  email: string;
  fullName?: string;
  role: string;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

export interface Directory {
  id: number;
  name: string;
  createdAt?: string;
}

export interface DirectoryFile {
  id: number;
  name: string;
  mimeType?: string;
  sizeBytes: number | string;
  storageStatus?: 'stored' | 'indexed' | 'error';
  indexedAt?: string | null;
  rawDeletedAt?: string | null;
  storageError?: string | null;
  createdAt?: string;
}

export interface DirectoryDetail extends Directory {
  files: DirectoryFile[];
}

export interface DirectoryIndexStatus {
  directoryId: number;
  status: 'not_indexed' | 'indexing' | 'completed' | 'stale' | 'error' | 'unavailable';
  progress: number;
  message: string;
  documentCount?: number;
  chunksWritten?: number;
  embeddingsWritten?: number;
  skippedFiles?: string[];
  updatedAt: string;
}

export interface IndexedChunk {
  id: string;
  documentId: string;
  relativePath: string;
  documentTitle: string;
  text: string;
  position: number;
  startChar: number;
  endChar: number;
  chunkType?: string | null;
  metadata: Record<string, unknown>;
  hasEmbedding: boolean;
}

export interface FileChunksResponse {
  directoryId: number;
  file: DirectoryFile;
  document?: {
    id: string;
    relativePath: string;
    title: string;
  };
  chunks: IndexedChunk[];
}

export interface ChatSession {
  id: number;
  title?: string;
  model?: string;
  temperature?: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface SessionFile extends DirectoryFile {
  directoryId: number;
}

export interface ResearchStep {
  id: number;
  stepId: string;
  status: 'pending' | 'running' | 'completed' | 'error' | string;
  title: string;
  preview?: string;
  details?: string;
  metadata?: Record<string, unknown>;
  createdAt?: string;
  completedAt?: string;
}

export interface Source {
  id: number;
  title: string;
  snippet?: string;
  url?: string;
  filePath?: string;
  page?: number;
  chunkId?: string;
  score?: number;
  createdAt?: string;
}

export interface LlmUsage {
  id: number;
  provider: string;
  model?: string;
  purpose: string;
  inputTokens: number;
  outputTokens: number;
  thinkingTokens: number;
  durationMs?: number;
  createdAt?: string;
}

export interface ChatMessageRecord {
  id: number;
  sessionId: number;
  role: 'user' | 'assistant';
  content: string;
  status: 'pending' | 'streaming' | 'completed' | 'error' | 'cancelled';
  createdAt?: string;
  updatedAt?: string;
  steps: ResearchStep[];
  sources: Source[];
  usage: LlmUsage[];
}

export type AgentEvent =
  | {type: 'message_created'; messageId: number}
  | {type: 'research_step'; step: ResearchStep}
  | {type: 'answer_delta'; text: string}
  | {type: 'source'; source: Source}
  | {type: 'done'; messageId: number; content: string; stats?: Record<string, unknown>}
  | {type: 'cancelled'; messageId: number}
  | {type: 'error'; message: string};
