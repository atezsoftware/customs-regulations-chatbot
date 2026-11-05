"use client";

import { useRef, useState } from "react";
import useSWR from "swr";
import { Button, Text } from "@opal/components";
import { SvgDownload, SvgPaperclip, SvgX } from "@opal/icons";
import { toast } from "@opal/layouts";
import type { ASv3SessionFile } from "@/lib/asv3/interfaces";
import {
  ASV3_SESSION_FILE_ACCEPT,
  deleteASv3SessionFile,
} from "@/lib/asv3/sessionFiles";
import { errorHandlingFetcher } from "@/lib/fetcher";

interface ASv3SessionAttachmentsProps {
  chatSessionId: string | null;
  draftFileIds: string[];
  revision: string;
  onUploadFiles: (files: File[]) => void;
  disabled?: boolean;
}

export default function ASv3SessionAttachments({
  chatSessionId,
  draftFileIds,
  revision,
  onUploadFiles,
  disabled = false,
}: ASv3SessionAttachmentsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [removingId, setRemovingId] = useState<string | null>(null);
  const { data, error, mutate } = useSWR<ASv3SessionFile[]>(
    chatSessionId
      ? [
          `/api/chat/sessions/${encodeURIComponent(chatSessionId)}/files`,
          revision,
        ]
      : null,
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

  return (
    <div className="flex flex-col gap-1 p-1">
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        multiple
        accept={ASV3_SESSION_FILE_ACCEPT}
        disabled={disabled}
        onChange={(event) => {
          const selected = Array.from(event.currentTarget.files ?? []);
          if (selected.length > 0) onUploadFiles(selected);
          event.currentTarget.value = "";
        }}
      />
      <Button
        icon={SvgPaperclip}
        prominence="secondary"
        size="sm"
        disabled={disabled}
        onClick={() => fileInputRef.current?.click()}
      >
        Dosya ekle
      </Button>
      {error && (
        <Text font="secondary-body" color="text-03">
          Oturum dosyaları yüklenemedi.
        </Text>
      )}
      {files.length > 0 && (
        <Text font="secondary-body" color="text-03">
          Bu oturumda kullanılan dosyalar
        </Text>
      )}
      <div className="flex flex-wrap gap-1">
        {files.map((file) => (
          <div
            key={file.id}
            className="flex items-center gap-1 rounded-08 border border-border-01 p-1"
          >
            <Text font="secondary-body">{file.name || file.id}</Text>
            <Button
              href={`/api/chat/files/${encodeURIComponent(file.id)}`}
              icon={SvgDownload}
              tooltip="Dosyayı indir"
              prominence="tertiary"
              size="sm"
            />
            <Button
              icon={SvgX}
              tooltip="Oturumdan kaldır"
              aria-label={`${file.name || file.id} dosyasını oturumdan kaldır`}
              prominence="tertiary"
              size="sm"
              disabled={removingId !== null}
              onClick={() => void removeFile(file.id)}
            />
          </div>
        ))}
      </div>
      {files.length > 0 && (
        <Text font="secondary-body" color="text-03">
          Dosyalar ASv3 Tek veya exp cropped seçildiğinde kullanılır.
        </Text>
      )}
    </div>
  );
}
