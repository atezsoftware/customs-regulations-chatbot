"use client";

import { useState } from "react";
import { Button, Text } from "@opal/components";
import type { LegislationFile } from "@/lib/legistations/interfaces";
import { fetchCatalogFile } from "@/lib/legistations/svc";

export default function LegislationSourceIdentity({
  file,
  onSelectFile,
}: {
  file: LegislationFile;
  onSelectFile?: (file: LegislationFile) => void;
}) {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lineage = file.source_lineage;
  if (!lineage) return null;

  async function openFile(id: string) {
    setLoading(id);
    setError(null);
    try {
      const linked = await fetchCatalogFile(id);
      onSelectFile?.(linked);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Bağlantılı dosya açılamadı."
      );
    } finally {
      setLoading(null);
    }
  }

  const related = [
    ...lineage.parents.map((item) => ({ ...item, label: "Ana dosya" })),
    ...lineage.attachments.map((item) => ({ ...item, label: "Ek" })),
  ];
  return (
    <div className="flex flex-col gap-1 break-words [overflow-wrap:anywhere]">
      {lineage.source.original_name !== file.name && (
        <Text font="secondary-body" color="text-03">
          {`Özgün dosya: ${lineage.source.original_name}`}
        </Text>
      )}
      {related.map((item) => {
        const title =
          !item.title.trim() || item.title === "Başlık belirlenemedi"
            ? item.source.original_name
            : item.title;
        const label = `${item.label}: ${title}`;
        return onSelectFile ? (
          <Button
            key={item.document_id}
            size="sm"
            prominence="tertiary"
            aria-label={label}
            disabled={loading !== null}
            onClick={() => void openFile(item.user_file_id)}
          >
            {loading === item.user_file_id ? "Açılıyor…" : label}
          </Button>
        ) : (
          <Text key={item.document_id} font="secondary-body" color="text-03">
            {label}
          </Text>
        );
      })}
      {error && (
        <div role="alert">
          <Text font="secondary-body">{error}</Text>
        </div>
      )}
    </div>
  );
}
