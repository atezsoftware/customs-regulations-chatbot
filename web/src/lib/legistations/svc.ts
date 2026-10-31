import type {
  CatalogPage,
  LegislationFile,
  LegislationFolder,
  LegislationSearchPage,
  LegistationsQuery,
} from "@/lib/legistations/interfaces";

export const CATALOG_PAGE_SIZE = 25;
export const FOLDER_PAGE_SIZE = 50;

async function catalogResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail =
      body && typeof body === "object" && "detail" in body ? body.detail : null;
    throw new Error(
      typeof detail === "string"
        ? detail
        : `İstek tamamlanamadı (${response.status}).`
    );
  }
  return response.json();
}

export function catalogFilesUrl(state: LegistationsQuery): string {
  const params = new URLSearchParams({
    q: state.query,
    offset: String(state.offset),
    limit: String(CATALOG_PAGE_SIZE),
    recursive: state.folder === "" ? "false" : "true",
  });
  if (state.folder !== null) params.set("folder", state.folder);
  return `/api/regulatory/catalog/files?${params}`;
}

export async function fetchCatalogFiles(
  state: LegistationsQuery
): Promise<CatalogPage<LegislationFile>> {
  return catalogResponse(await fetch(catalogFilesUrl(state)));
}

export function catalogFileUrl(fileId: string): string {
  return `/api/regulatory/catalog/files/${encodeURIComponent(fileId)}`;
}

export async function fetchCatalogFile(
  fileId: string
): Promise<LegislationFile> {
  return catalogResponse(await fetch(catalogFileUrl(fileId)));
}

export async function indexCatalogFile(
  fileId: string
): Promise<LegislationFile> {
  return catalogResponse(
    await fetch(`${catalogFileUrl(fileId)}/index`, { method: "POST" })
  );
}

export function catalogFoldersUrl(
  parent: string,
  query: string,
  offset: number
): string {
  const params = new URLSearchParams({
    parent,
    q: query,
    offset: String(offset),
    limit: String(FOLDER_PAGE_SIZE),
  });
  return `/api/regulatory/catalog/folders?${params}`;
}

export async function fetchCatalogFolders(
  parent: string,
  query: string,
  offset: number
): Promise<CatalogPage<LegislationFolder>> {
  return catalogResponse(await fetch(catalogFoldersUrl(parent, query, offset)));
}

export async function searchCatalogChunks(
  state: LegistationsQuery
): Promise<LegislationSearchPage> {
  return catalogResponse(
    await fetch("/api/regulatory/catalog/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: state.query,
        mode: state.mode,
        offset: state.offset,
        limit: CATALOG_PAGE_SIZE,
      }),
    })
  );
}
