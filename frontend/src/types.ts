export interface SafeUser {
  id: number;
  email: string;
  fullName?: string;
  role: string;
  uploadsEnabled: boolean;
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
  storageStatus?: 'stored' | 'chunked' | 'indexed' | 'error';
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
  status:
    | 'not_indexed'
    | 'chunking'
    | 'chunked'
    | 'indexing'
    | 'completed'
    | 'stale'
    | 'error'
    | 'unavailable';
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
  chunkType?: string | null;
  headingPath: string[];
  metadata: Record<string, unknown>;
  hasEmbedding: boolean;
  source: 'indexed' | 'amendment';
  status: 'active' | 'superseded' | 'expired';
  effectiveStartDate: string | null;
  effectiveEndDate: string | null;
  supersedesChunkId: string | null;
  supersededByChunkId: string | null;
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
  /** All-time sum of input+output+thinking tokens across this session's LLM calls. */
  totalTokens?: number;
  /** Fraction (0-1) of the model's context window used by the most recent turn's last call, or null if no turn has completed yet. */
  lastContextUsageRatio?: number | null;
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
  // Set once core-api's `start` event arrives. Kept around after an
  // 'error'/'cancelled' status so the UI can offer to resume this exact
  // run (via ?resumeRunId=) instead of only "Regenerate" (a brand-new run
  // that throws away whatever the interrupted run had already gathered).
  runId?: string;
  // True when a 'completed' message hit core-api's step-budget safety net
  // rather than a real conclusion — not an error, but still worth offering
  // "Continue" for instead of treating it as a normal finished answer.
  incomplete?: boolean;
}

export type AgentEvent =
  | {type: 'message_created'; messageId: number}
  | {type: 'run_started'; runId: string; resumed: boolean}
  | {type: 'research_step'; step: ResearchStep}
  | {type: 'answer_delta'; text: string}
  | {type: 'source'; source: Source}
  | {
      type: 'done';
      messageId: number;
      content: string;
      stats?: Record<string, unknown>;
      incomplete?: boolean;
    }
  | {type: 'cancelled'; messageId: number}
  | {type: 'error'; message: string; runId?: string};

export type UsageRange = '7d' | '30d' | '90d' | 'all';

export interface UsageAnalytics {
  range: UsageRange;
  since?: string | null;
  totals: {
    calls: number;
    sessions: number;
    inputTokens: number;
    outputTokens: number;
    thinkingTokens: number;
    totalTokens: number;
    avgDurationMs?: number | null;
  };
  daily: Array<{
    day: string;
    inputTokens: number;
    outputTokens: number;
    thinkingTokens: number;
    totalTokens: number;
    calls: number;
  }>;
  topSessions: Array<{
    sessionId: number;
    title: string;
    updatedAt?: string;
    calls: number;
    inputTokens: number;
    outputTokens: number;
    thinkingTokens: number;
    totalTokens: number;
  }>;
  models: Array<{
    provider: string;
    model: string;
    calls: number;
    totalTokens: number;
  }>;
}

export interface AdminSupportUser {
  id: number;
  email: string;
  fullName?: string;
  role: string;
}

export interface AdminSupportSession {
  id: number;
  title: string;
  createdAt?: string;
  updatedAt?: string;
  user: AdminSupportUser;
  messageCount: number;
  totalTokens: number;
  lastMessageAt?: string;
  lastMessage?: {
    role?: string | null;
    status?: string | null;
    preview: string;
  };
}

export interface AdminSupportSessionsResponse {
  sessions: AdminSupportSession[];
}

export interface AdminSupportSessionDetail {
  session: {
    id: number;
    title: string;
    createdAt?: string;
    updatedAt?: string;
    user: AdminSupportUser;
  };
  messages: ChatMessageRecord[];
}

export interface AmendmentProposal {
  id: string;
  batchId: string;
  instructionIndex: number;
  instructionText: string;
  oldChunkId: string | null;
  oldChunkSnapshot: Record<string, unknown>;
  newChunkDraft: Record<string, unknown>;
  matchConfidence: number | null;
  matchRationale: string | null;
  dateRationale: string | null;
  status: 'pending' | 'approved' | 'rejected';
  appliedNewChunkId: string | null;
  decidedBy: string | null;
  decidedAt: string | null;
  createdAt: string;
  updatedAt: string;
  duplicateTarget: boolean;
}

export interface AmendmentBatch {
  id: string;
  corpusId: string;
  rawText: string;
  referenceDate: string | null;
  status: 'analyzing' | 'analyzed' | 'failed';
  errorMessage: string | null;
  createdBy: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface AnalyzeAmendmentResult {
  batchId: string;
  referenceDate: string | null;
  proposals: AmendmentProposal[];
  unmatchedInstructions: string[];
}

export interface ApproveAmendmentResult {
  applied: AmendmentProposal[];
  failed: Array<{proposalId: string; reason: string}>;
}
