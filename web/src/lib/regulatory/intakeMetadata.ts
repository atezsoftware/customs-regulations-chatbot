export const intakeLegalMetadataFields = [
  ["document_title", "Document title", "text"],
  ["document_type", "Document type", "text"],
  ["document_number", "Document number", "text"],
  ["document_date", "Document date", "text"],
  ["publication_date", "Publication date", "text"],
  ["publication_number", "Publication number", "text"],
  ["article_no", "Article number", "text"],
  ["article_title", "Article title", "text"],
  ["paragraph_no", "Paragraph number", "text"],
  ["clause_label", "Clause label", "text"],
  ["subclause_label", "Subclause label", "text"],
  ["appendix_label", "Appendix label", "text"],
  ["table_index", "Table index", "integer"],
  ["table_row_index", "Table row index", "integer"],
  ["section_title", "Section title", "text"],
  ["annex_title", "Annex title", "text"],
] as const;

export interface IntakeMetadataEdits {
  patch: Record<string, string | number | null>;
  dirty: boolean;
  error: string | null;
}

export function metadataRecord(value: unknown): Record<string, unknown> {
  return value && !Array.isArray(value) && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

export function intakeMetadataInput(value: unknown): string {
  if (value === null || value === undefined) return "";
  return typeof value === "object"
    ? JSON.stringify(value, null, 2)
    : String(value);
}

export function intakeMetadataInputs(
  metadata: Record<string, unknown>
): Record<string, string> {
  return Object.fromEntries(
    intakeLegalMetadataFields.map(([key]) => [
      key,
      intakeMetadataInput(metadata[key]),
    ])
  );
}

export function intakeMetadataEdits(
  metadata: Record<string, unknown>,
  inputs: Record<string, string>
): IntakeMetadataEdits {
  const patch: IntakeMetadataEdits["patch"] = {};
  let dirty = false;
  for (const [key, label, type] of intakeLegalMetadataFields) {
    const value = inputs[key] ?? "";
    if (value === intakeMetadataInput(metadata[key])) continue;
    dirty = true;
    if (!value.trim()) patch[key] = null;
    else if (type === "integer") {
      if (!/^-?\d+$/.test(value.trim()) || !Number.isSafeInteger(Number(value)))
        return { patch, dirty, error: `${label} must be a whole number.` };
      patch[key] = Number(value);
    } else patch[key] = value;
  }
  return { patch, dirty, error: null };
}
