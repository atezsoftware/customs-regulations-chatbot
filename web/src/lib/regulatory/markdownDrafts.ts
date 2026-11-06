import type { AmendmentSourcePackage } from "@/lib/regulatory/amendments";

export interface MarkdownDraftDocument {
  id: string;
  display_name: string;
  parent_document_id: string | null;
  byte_count: number;
  sha256: string;
}

export interface MarkdownDraft {
  id: string;
  revision: number;
  status: "draft" | "prepared";
  documents: MarkdownDraftDocument[];
  source_package_id: string | null;
  updated_at: string;
}

const base = "/api/regulatory/document-intake/markdown-drafts";

export class MarkdownDraftRequestError extends Error {
  constructor(
    public readonly status: number,
    detail?: unknown
  ) {
    const turkishDetail =
      typeof detail === "string" &&
      (/[çğıöşüÇĞİÖŞÜ]/.test(detail) ||
        detail.startsWith("Her Markdown belgesi") ||
        detail.startsWith("En fazla "));
    super(
      turkishDetail
        ? detail
        : status === 409
          ? "Taslak başka bir işlemde değişti. Güncel taslağı yükleyip yeniden deneyin."
          : status === 413
            ? "Dosyalar çok büyük. Daha küçük Markdown dosyaları seçin."
            : status === 422 || status === 400
              ? "Dosyalar eklenemedi. Dosyaların .md uzantılı, UTF-8 biçiminde ve birbirinden farklı olduğunu kontrol edin."
              : status === 403 || status === 404
                ? "Bu taslağa erişilemiyor. Taslak listesini yenileyin."
                : "İşlem tamamlanamadı. Bağlantınızı kontrol edip yeniden deneyin."
    );
    this.name = "MarkdownDraftRequestError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, init);
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail =
      body && typeof body === "object" && "detail" in body ? body.detail : null;
    throw new MarkdownDraftRequestError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export function listMarkdownDrafts(): Promise<MarkdownDraft[]> {
  return request("");
}

export function getMarkdownDraft(id: string): Promise<MarkdownDraft> {
  return request(`/${encodeURIComponent(id)}`);
}

export function createMarkdownDraft(key: string): Promise<MarkdownDraft> {
  return request("", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ idempotency_key: key }),
  });
}

export function uploadMarkdownDraftDocuments(
  draft: MarkdownDraft,
  files: File[],
  key: string,
  parentId: string | null
): Promise<MarkdownDraft> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  body.append("idempotency_key", key);
  body.append("expected_revision", String(draft.revision));
  if (parentId) body.append("parent_document_id", parentId);
  return request(`/${encodeURIComponent(draft.id)}/documents`, {
    method: "POST",
    body,
  });
}

export function removeMarkdownDraftDocument(
  draft: MarkdownDraft,
  documentId: string
): Promise<MarkdownDraft> {
  return request(
    `/${encodeURIComponent(draft.id)}/documents/${encodeURIComponent(documentId)}?expected_revision=${draft.revision}`,
    { method: "DELETE" }
  );
}

export function prepareMarkdownDraft(
  draft: MarkdownDraft
): Promise<AmendmentSourcePackage> {
  return request(`/${encodeURIComponent(draft.id)}/prepare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: draft.revision }),
  });
}
