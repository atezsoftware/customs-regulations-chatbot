import type {
  ASv3SessionFile,
  ASv3SessionFileUpload,
} from "@/lib/asv3/interfaces";
import type { ProjectFile } from "@/lib/projects/types";
import { UserFileStatus } from "@/lib/projects/types";

export const ASV3_SESSION_FILE_ACCEPT = ".pdf,.docx,.xlsx,.csv,.txt,.md";

export async function deleteASv3SessionFile(
  chatSessionId: string | null,
  fileId: string
): Promise<void> {
  const path = chatSessionId
    ? `/api/chat/sessions/${encodeURIComponent(chatSessionId)}/files/${encodeURIComponent(fileId)}`
    : `/api/chat/files/${encodeURIComponent(fileId)}`;
  const response = await fetch(path, { method: "DELETE" });
  if (!response.ok) throw new Error("Dosya session'dan kaldırılamadı.");
}

export async function uploadASv3SessionFiles(
  files: File[],
  chatSessionId?: string | null
): Promise<ASv3SessionFileUpload> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  if (chatSessionId) body.append("chat_session_id", chatSessionId);
  const response = await fetch("/api/chat/files/upload", {
    method: "POST",
    body,
  });
  if (!response.ok) {
    const error: { detail?: string } = await response.json();
    throw new Error(error.detail || "Dosyalar yüklenemedi.");
  }
  return response.json();
}

export function asv3SessionFileToProjectFile(
  file: ASv3SessionFile,
  status = UserFileStatus.COMPLETED
): ProjectFile {
  const now = new Date().toISOString();
  return {
    id: file.id,
    file_id: file.id,
    name: file.name || "Dosya",
    project_id: null,
    user_id: null,
    created_at: now,
    last_accessed_at: now,
    status,
    file_type: "application/octet-stream",
    chat_file_type: file.type,
    token_count: file.token_count,
    chunk_count: null,
    is_session_attachment: true,
  };
}
