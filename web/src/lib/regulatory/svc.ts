import type { ProjectFile } from "@/lib/projects/types";
import type {
  RegulatoryChunk,
  RegulatoryChunkUpdate,
} from "@/lib/regulatory/interfaces";

const handleRequestError = (action: string, response: Response): never => {
  throw new Error(`${action} failed (Status: ${response.status})`);
};

export async function fetchChunksForFile(
  userFileId: string
): Promise<RegulatoryChunk[]> {
  const response = await fetch(`/api/regulatory/files/${userFileId}/chunks`);
  if (!response.ok) {
    handleRequestError("Fetch chunks", response);
  }
  return response.json();
}

export async function patchChunk(
  chunkId: string,
  update: RegulatoryChunkUpdate
): Promise<RegulatoryChunk> {
  const response = await fetch(`/api/regulatory/chunks/${chunkId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!response.ok) {
    handleRequestError("Update chunk", response);
  }
  return response.json();
}

export async function renameUserFile(
  userFileId: string,
  name: string
): Promise<ProjectFile> {
  const response = await fetch(`/api/regulatory/files/${userFileId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) {
    handleRequestError("Rename file", response);
  }
  return response.json();
}

/**
 * The same bytes back both uses: the browser renders them inline and saves them
 * unchanged, so there is no separate download route.
 */
export function documentPdfUrl(userFileId: string): string {
  return `/api/regulatory/files/${encodeURIComponent(userFileId)}/pdf`;
}

export function chunkPdfUrl(chunkId: string): string {
  return `/api/regulatory/chunks/${encodeURIComponent(chunkId)}/pdf`;
}
