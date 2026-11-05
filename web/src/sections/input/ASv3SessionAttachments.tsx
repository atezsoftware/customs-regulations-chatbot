"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button, Text } from "@opal/components";
import { SvgDownload, SvgX } from "@opal/icons";
import { toast } from "@opal/layouts";
import type { ASv3SessionFile } from "@/lib/asv3/interfaces";
import { deleteASv3SessionFile } from "@/lib/asv3/sessionFiles";
import { errorHandlingFetcher } from "@/lib/fetcher";

interface ASv3SessionAttachmentsProps {
  chatSessionId: string | null;
  draftFileIds: string[];
  revision: string;
  onUseDraftFile: (file: ASv3SessionFile) => void;
}

export default function ASv3SessionAttachments({
  chatSessionId,
  draftFileIds,
  revision,
  onUseDraftFile,
}: ASv3SessionAttachmentsProps) {
  const [removingId, setRemovingId] = useState<string | null>(null);
  const { data, error, mutate } = useSWR<ASv3SessionFile[]>(
    [
      chatSessionId
        ? `/api/chat/sessions/${encodeURIComponent(chatSessionId)}/files`
        : "/api/chat/files",
      revision,
    ],
    ([url]: [string, string]) => errorHandlingFetcher<ASv3SessionFile[]>(url)
  );
  const files = data?.filter((file) => !draftFileIds.includes(file.id)) ?? [];

  async function removeFile(fileId: string) {
    setRemovingId(fileId);
    try {
      await deleteASv3SessionFile(chatSessionId, fileId);
      await mutate(
        data?.filter((file) => file.id !== fileId),
        false
      );
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Dosya kaldırılamadı."
      );
    } finally {
      setRemovingId(null);
    }
  }

  if (error) {
    return (
      <Text font="secondary-body" color="text-03">
        {chatSessionId
          ? "Oturum dosyaları yüklenemedi."
          : "Gönderilmemiş dosyalar yüklenemedi."}
      </Text>
    );
  }
  if (files.length === 0) return null;

  return (
    <div className="flex flex-col gap-1 p-1">
      <Text font="secondary-body" color="text-03">
        {chatSessionId
          ? "Bu oturumda kullanılan dosyalar"
          : "Gönderilmemiş dosyalar"}
      </Text>
      <div className="flex flex-wrap gap-1">
        {files.map((file) => (
          <div
            key={file.id}
            className="flex items-center gap-1 rounded-08 border border-border-01 p-1"
          >
            <Text font="secondary-body">{file.name || file.id}</Text>
            {!chatSessionId && (
              <Button
                prominence="secondary"
                size="sm"
                disabled={removingId !== null}
                onClick={() => onUseDraftFile(file)}
              >
                Bu soruya ekle
              </Button>
            )}
            <Button
              href={`/api/chat/files/${encodeURIComponent(file.id)}`}
              icon={SvgDownload}
              tooltip="Dosyayı indir"
              prominence="tertiary"
              size="sm"
            />
            <Button
              icon={SvgX}
              tooltip={
                chatSessionId ? "Oturumdan kaldır" : "Taslak dosyayı sil"
              }
              aria-label={`${file.name || file.id} dosyasını ${chatSessionId ? "oturumdan kaldır" : "sil"}`}
              prominence="tertiary"
              size="sm"
              disabled={removingId !== null}
              onClick={() => void removeFile(file.id)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
