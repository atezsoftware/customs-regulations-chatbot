import fs from 'fs/promises';
import path from 'path';

const STORAGE_ROOT = process.env.STORAGE_ROOT ?? './storage';
const DEFAULT_CORE_URL = 'http://127.0.0.1:8000';

export type DirectoryIndexState =
  | 'not_indexed'
  | 'indexing'
  | 'completed'
  | 'stale'
  | 'error'
  | 'unavailable';

export interface DirectoryIndexStatus {
  directoryId: number;
  status: DirectoryIndexState;
  progress: number;
  message: string;
  documentCount?: number;
  chunksWritten?: number;
  embeddingsWritten?: number;
  skippedFiles?: string[];
  updatedAt: string;
}

export interface IndexableDirectoryFile {
  id?: number;
  originalName: string;
  storedPath: string;
  storageStatus?: string;
}

export interface SkippedDirectoryFile {
  file?: IndexableDirectoryFile;
  label: string;
  reason: string;
}

export interface DirectoryIndexCompletion {
  directoryId: number;
  corpusKey: string;
  indexedFiles: IndexableDirectoryFile[];
  skippedFiles: SkippedDirectoryFile[];
  result: Record<string, unknown>;
}

export interface StartDirectoryIndexOptions {
  onCompleted?: (completion: DirectoryIndexCompletion) => Promise<void>;
}

const activeJobs = new Map<number, DirectoryIndexStatus>();

export function startDirectoryIndex(
  directoryId: number,
  files: IndexableDirectoryFile[],
  options: StartDirectoryIndexOptions = {},
): DirectoryIndexStatus {
  const existing = activeJobs.get(directoryId);
  if (existing?.status === 'indexing') return existing;

  console.info(`[directory-index] starting directory=${directoryId}`);
  const started: DirectoryIndexStatus = {
    directoryId,
    status: 'indexing',
    progress: 10,
    message: 'Indexing started. Generating embeddings after chunking.',
    updatedAt: new Date().toISOString(),
  };
  activeJobs.set(directoryId, started);

  void runDirectoryIndex(directoryId, files, options).catch(error => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[directory-index] failed directory=${directoryId}: ${message}`);
    activeJobs.set(directoryId, {
      directoryId,
      status: 'error',
      progress: 0,
      message,
      updatedAt: new Date().toISOString(),
    });
  });

  return started;
}

export async function getDirectoryIndexStatus(directoryId: number): Promise<DirectoryIndexStatus> {
  const active = activeJobs.get(directoryId);
  if (active?.status === 'indexing' || active?.status === 'error') return active;

  const endpoint = `${resolveCoreRestUrl()}/api/index/status`;
  const headers: Record<string, string> = {};
  if (process.env.CORE_INTERNAL_TOKEN) {
    headers['X-Internal-Token'] = process.env.CORE_INTERNAL_TOKEN;
  }

  try {
    const params = new URLSearchParams({folder: virtualCorpusKey(directoryId)});
    const databaseUrl = resolveDatabaseUrl();
    if (databaseUrl) params.set('database_url', databaseUrl);
    const res = await fetch(`${endpoint}?${params.toString()}`, {headers});
    if (!res.ok) {
      return {
        directoryId,
        status: 'unavailable',
        progress: 0,
        message: `Core index status failed (${res.status}).`,
        updatedAt: new Date().toISOString(),
      };
    }

    const data = (await res.json()) as Record<string, unknown>;
    if (!data.indexed) {
      return {
        directoryId,
        status: 'not_indexed',
        progress: 0,
        message: 'Not indexed yet.',
        updatedAt: new Date().toISOString(),
      };
    }

    const fresh = Boolean(data.fresh);
    const status: DirectoryIndexStatus = {
      directoryId,
      status: fresh ? 'completed' : 'stale',
      progress: fresh ? 100 : 65,
      message: fresh ? 'Index is complete and fresh.' : 'Files changed after indexing.',
      documentCount: numberOrUndefined(data.document_count),
      updatedAt: new Date().toISOString(),
    };
    activeJobs.set(directoryId, status);
    return status;
  } catch (error) {
    return {
      directoryId,
      status: 'unavailable',
      progress: 0,
      message: error instanceof Error ? error.message : String(error),
      updatedAt: new Date().toISOString(),
    };
  }
}

async function runDirectoryIndex(
  directoryId: number,
  files: IndexableDirectoryFile[],
  options: StartDirectoryIndexOptions,
): Promise<void> {
  console.info(`[directory-index] parsing/chunking directory=${directoryId}`);
  activeJobs.set(directoryId, {
    directoryId,
    status: 'indexing',
    progress: 35,
    message: 'Parsing files and creating regulatory chunks.',
    updatedAt: new Date().toISOString(),
  });

  const manifest = await buildDirectoryIndexManifest(directoryId, files);
  if (!manifest.indexedFiles.length) {
    throw new Error(
      `No valid indexable files found. Skipped: ${formatSkippedFiles(manifest.skippedFiles).join('; ') || 'none'}`,
    );
  }

  const result = await triggerDirectoryIndex(
    directoryId,
    manifest.corpusKey,
    manifest.documents,
  );
  await options.onCompleted?.({
    directoryId,
    corpusKey: manifest.corpusKey,
    indexedFiles: manifest.indexedFiles,
    skippedFiles: manifest.skippedFiles,
    result,
  });
  console.info(
    `[directory-index] completed directory=${directoryId} ` +
      `documents=${String(result.active_documents ?? 'n/a')} ` +
      `chunks=${String(result.chunks_written ?? 'n/a')} ` +
      `embeddings=${String(result.embeddings_written ?? 'n/a')}`,
  );
  const completed: DirectoryIndexStatus = {
    directoryId,
    status: 'completed',
    progress: 100,
    message: manifest.skippedFiles.length
      ? `Index and embeddings are ready. Skipped ${manifest.skippedFiles.length} invalid file(s).`
      : 'Index and embeddings are ready.',
    documentCount: numberOrUndefined(result.active_documents),
    chunksWritten: numberOrUndefined(result.chunks_written),
    embeddingsWritten: numberOrUndefined(result.embeddings_written),
    skippedFiles: formatSkippedFiles(manifest.skippedFiles),
    updatedAt: new Date().toISOString(),
  };
  activeJobs.set(directoryId, completed);
}

export async function triggerDirectoryIndex(
  directoryId: number,
  corpusKey: string,
  documents: IndexDocumentManifest[],
): Promise<Record<string, unknown>> {
  const endpoint = `${resolveCoreRestUrl()}/api/index`;
  const headers: Record<string, string> = {'Content-Type': 'application/json'};
  if (process.env.CORE_INTERNAL_TOKEN) {
    headers['X-Internal-Token'] = process.env.CORE_INTERNAL_TOKEN;
  }

  const body = {
    folder: corpusKey,
    corpus_key: corpusKey,
    documents,
    database_url: resolveDatabaseUrl(),
    with_embeddings: true,
  };

  console.info(
    `[directory-index] POST ${endpoint} directory=${directoryId} corpus=${body.corpus_key} documents=${documents.length}`,
  );

  let res: Response;
  try {
    res = await fetch(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new Error(
      `Core is not reachable at ${endpoint}. Start core with ` +
        '`scripts/run.sh --env dev --apps core` or run the full stack. ' +
        `Original error: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(
      `Core indexing failed (${res.status} ${res.statusText}): ${body || 'empty response'}`,
    );
  }

  const responseBody = (await res.json()) as Record<string, unknown>;
  if (responseBody.error) {
    throw new Error(`Core indexing returned error: ${String(responseBody.error)}`);
  }
  return responseBody;
}

function resolveCoreRestUrl(): string {
  const configured = process.env.CORE_INTERNAL_URL ?? DEFAULT_CORE_URL;
  const base = configured
    .replace(/\/ws\/explore$/, '')
    .replace(/^ws:/, 'http:')
    .replace(/^wss:/, 'https:')
    .replace(/\/$/, '');
  return base;
}

function resolveDatabaseUrl(): string | undefined {
  if (process.env.DATABASE_URL) return process.env.DATABASE_URL;
  const host = process.env.DB_HOST;
  const user = process.env.DB_USER;
  const password = process.env.DB_PASSWORD;
  const database = process.env.DB_NAME;
  if (!host || !user || !database) return undefined;
  const port = process.env.DB_PORT ?? '5432';
  return `postgresql://${encodeURIComponent(user)}:${encodeURIComponent(
    password ?? '',
  )}@${host}:${port}/${database}`;
}

export function virtualCorpusKey(directoryId: number): string {
  return `/__customs_regulations__/directories/${directoryId}`;
}

interface IndexDocumentManifest {
  file_path: string;
  relative_path: string;
  display_name: string;
  logical_path: string;
}

async function buildDirectoryIndexManifest(
  directoryId: number,
  files: IndexableDirectoryFile[],
): Promise<{
  corpusKey: string;
  documents: IndexDocumentManifest[];
  indexedFiles: IndexableDirectoryFile[];
  skippedFiles: SkippedDirectoryFile[];
}> {
  const root = path.resolve(STORAGE_ROOT);

  const indexedFiles: IndexableDirectoryFile[] = [];
  const documents: IndexDocumentManifest[] = [];
  const skippedFiles: SkippedDirectoryFile[] = [];
  for (const file of files) {
    const target = path.resolve(file.storedPath);
    const label = file.originalName || path.basename(file.storedPath);
    if (!target.startsWith(root + path.sep)) {
      skippedFiles.push({file, label, reason: 'stored path is outside STORAGE_ROOT'});
      continue;
    }

    let rawFileExists = false;
    try {
      const stat = await fs.stat(target);
      if (!stat.isFile()) {
        skippedFiles.push({file, label, reason: 'stored path is not a file'});
        continue;
      }
      rawFileExists = true;
    } catch {
      rawFileExists = false;
    }

    if (!rawFileExists) {
      skippedFiles.push({file, label, reason: 'stored file is missing'});
      continue;
    }

    if (path.extname(label).toLowerCase() === '.docx' && !(await isZipContainer(target))) {
      skippedFiles.push({file, label, reason: 'invalid DOCX file'});
      continue;
    }

    const relativePath = `${file.id ?? indexedFiles.length + 1}-${safeSegment(label)}`;
    documents.push({
      file_path: target,
      relative_path: relativePath,
      display_name: label,
      logical_path: label,
    });
    indexedFiles.push(file);
  }

  return {
    corpusKey: virtualCorpusKey(directoryId),
    documents,
    indexedFiles,
    skippedFiles,
  };
}

async function isZipContainer(filePath: string): Promise<boolean> {
  const handle = await fs.open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(4);
    const {bytesRead} = await handle.read(buffer, 0, 4, 0);
    return bytesRead >= 2 && buffer[0] === 0x50 && buffer[1] === 0x4b;
  } finally {
    await handle.close();
  }
}

function safeSegment(value: string): string {
  const cleaned = value.replace(/[<>:"/\\|?*\x00-\x1F]+/g, '_').trim();
  return cleaned || 'file';
}

function formatSkippedFiles(skippedFiles: SkippedDirectoryFile[]): string[] {
  return skippedFiles.map(file => `${file.label}: ${file.reason}`);
}

function numberOrUndefined(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}
