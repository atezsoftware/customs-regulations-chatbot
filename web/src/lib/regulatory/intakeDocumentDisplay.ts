import type { IntakeDocument } from "@/lib/regulatory/documentIntake";

export function intakeLegalTitleUnresolved(document: IntakeDocument): boolean {
  return document.display_name === "Başlık belirlenemedi";
}

export function intakeDocumentCaption(document: IntakeDocument): string {
  return intakeLegalTitleUnresolved(document)
    ? document.source_display_name || document.display_name
    : document.display_name;
}
