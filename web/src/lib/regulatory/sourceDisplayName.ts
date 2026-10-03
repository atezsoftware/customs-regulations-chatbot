import { OnyxDocument } from "@/lib/search/interfaces";

const FILE_EXTENSION = /\.(?:docx?|pdf|md|txt|html?|rtf|odt|xlsx?|csv|pptx?)$/i;
const NUMBERED_TITLE = /^[^/\\]*\b\d{4}\/\d+\b[^/\\]*$/;

function headingParts(document: OnyxDocument): string[] {
  const path = document.metadata?.regulatory_heading_path;
  return (
    Array.isArray(path)
      ? path
      : typeof path === "string"
        ? path.split(" > ")
        : []
  )
    .map((part) => part.trim())
    .filter(Boolean);
}

function readableBasename(identifier: string): string {
  const title = identifier.split(" — ")[0]?.trim() || identifier.trim();
  // A literal year/number in a human title is not a folder separator.
  const isNumberedTitle = NUMBERED_TITLE.test(title);
  const basename = isNumberedTitle
    ? title
    : title.split(/[/\\]/).pop() || title;
  return basename
    .replace(FILE_EXTENSION, "")
    .replace(/_+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function getRegulatorySourceDisplayName(
  document: OnyxDocument
): string | null {
  const preferred = document.metadata?.asv3_source_display_name;
  if (typeof preferred === "string" && preferred.trim())
    return preferred.trim();

  const headings = headingParts(document);
  const isRegulatory =
    headings.length > 0 ||
    Boolean(document.metadata?.regulatory_chunk_id) ||
    /\/api\/asv3\/citation\//.test(document.citation_preview_url || "") ||
    /\/api\/asv3\/citation\//.test(document.link);
  if (!isRegulatory) return null;

  if (document.metadata?.document_type === "genelge") {
    const number = headings
      .map((heading) => heading.match(/^\(?\s*(\d{4})\/(\d+)\s*\)?$/))
      .find((match) => match !== null);
    if (number) return `${number[1]}/${number[2]} sayılı Genelge`;
  }

  const root = headings[0];
  if (
    root &&
    !root.includes("_") &&
    !FILE_EXTENSION.test(root) &&
    !root.includes("\\") &&
    (!root.includes("/") || NUMBERED_TITLE.test(root)) &&
    /(?:kanunu?|yönetmeli[ğk]i?|tebli[ğg]i?|genelge(?:si)?|sirküler(?:i)?|karar(?:name|ı)?|sözleşme(?:si)?|protokol(?:ü)?)(?=\s|$|[(:])/.test(
      root.toLocaleLowerCase("tr")
    )
  ) {
    return root;
  }

  return readableBasename(document.semantic_identifier || "") || null;
}

export function getDocumentSourceDisplayName(document: OnyxDocument): string {
  return (
    getRegulatorySourceDisplayName(document) ||
    document.semantic_identifier ||
    ""
  );
}
