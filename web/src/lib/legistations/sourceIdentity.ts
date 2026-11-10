import type { LegislationFile } from "@/lib/legistations/interfaces";

export function legislationDisplayName(file: LegislationFile): string {
  if (file.display_name?.trim()) return file.display_name;
  return !file.name.trim() || file.name === "Başlık belirlenemedi"
    ? file.source_lineage?.source.original_name || file.name
    : file.name;
}
