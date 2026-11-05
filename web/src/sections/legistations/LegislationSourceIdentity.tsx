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

  const linked = [
    ...(lineage?.parents ?? []).map((item) => ({
      user_file_id: item.user_file_id,
      title: item.title,
      original_name: item.source.original_name,
      label: "Ana dosya",
    })),
    ...(file.reviewed_relations?.parents ?? []).map((item) => ({
      ...item,
      label: "Ana dosya",
    })),
    ...(lineage?.attachments ?? []).map((item) => ({
      user_file_id: item.user_file_id,
      title: item.title,
      original_name: item.source.original_name,
      label: "Ek",
    })),
    ...(file.reviewed_relations?.attachments ?? []).map((item) => ({
      ...item,
      label: "Ek",
    })),
  ];
  const related = Array.from(
    new Map(linked.map((item) => [item.user_file_id, item])).values()
  );
  if (!lineage && related.length === 0) return null;
  return (
    <div className="flex flex-col gap-1 break-words [overflow-wrap:anywhere]">
      {lineage && lineage.source.original_name !== file.name && (
        <Text font="secondary-body" color="text-03">
          {`Özgün dosya: ${lineage.source.original_name}`}
        </Text>
      )}
      {related.map((item) => {
        const title =
          !item.title.trim() || item.title === "Başlık belirlenemedi"
            ? item.original_name || item.title
            : item.title;
        const label = `${item.label}: ${title}`;
        return onSelectFile ? (
          <div
            key={item.user_file_id}
            className="flex min-w-0 flex-col items-start gap-1"
          >
            <Text font="secondary-body" color="text-03">
              {label}
            </Text>
            <Button
              size="sm"
              prominence="tertiary"
              aria-label={`${item.label === "Ana dosya" ? "Ana dosyayı aç" : "Eki aç"}: ${title}`}
              disabled={loading !== null}
              onClick={() => void openFile(item.user_file_id)}
            >
              {loading === item.user_file_id
                ? "Açılıyor…"
                : item.label === "Ana dosya"
                  ? "Ana dosyayı aç"
                  : "Eki aç"}
            </Button>
          </div>
        ) : (
          <Text key={item.user_file_id} font="secondary-body" color="text-03">
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
