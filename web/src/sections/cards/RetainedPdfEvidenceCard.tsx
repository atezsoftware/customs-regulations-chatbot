"use client";

import useSWR from "swr";
import { Button, Tag, Text } from "@opal/components";
import { getDocumentIntakeCapabilities } from "@/lib/regulatory/documentIntake";
import {
  fetchRetainedPdfEvidence,
  retainedPdfEvidenceUrl,
} from "@/lib/regulatory/svc";
import { formatBytes } from "@/lib/utils";

interface RetainedPdfEvidenceCardProps {
  fileId: string;
}

export default function RetainedPdfEvidenceCard({
  fileId,
}: RetainedPdfEvidenceCardProps) {
  const { data: capability, error: capabilityError } = useSWR(
    "document-intake-capability",
    getDocumentIntakeCapabilities
  );
  const documentSetId = capability?.enabled
    ? capability.target_document_set_id
    : null;
  const { data, error, isLoading } = useSWR(
    documentSetId === null
      ? null
      : (["retained-pdf-evidence", fileId, documentSetId] as const),
    ([, id, setId]) => fetchRetainedPdfEvidence(id, setId)
  );
  if (capabilityError || error) {
    return (
      <Text font="secondary-body" color="status-error-05">
        Source PDFs could not be loaded.
      </Text>
    );
  }
  if (isLoading) {
    return <Text font="secondary-body">Loading source PDFs…</Text>;
  }
  if (!data || data.user_file_id !== fileId || data.items.length === 0)
    return null;

  return (
    <section
      aria-label="Original source PDFs"
      className="flex flex-col gap-2 rounded-12 border border-border-02 p-3"
    >
      <Text as="h3" font="main-ui-action">
        Original source PDFs
      </Text>
      {data.items.map((item) => {
        const historical = item.relation === "historical_rendition";
        const current =
          item.canonical_sha256 === data.canonical_sha256 &&
          item.canonical_generation_hash === data.canonical_generation_hash;
        return (
          <div key={item.id} className="flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Text font="secondary-action" as="p">
                {item.original_filename}
              </Text>
              <Tag
                title={
                  historical
                    ? "Historical source"
                    : current
                      ? "Equivalent source"
                      : "Earlier retained source"
                }
              />
              <Text font="secondary-body" color="text-03">
                {formatBytes(item.byte_count)}
              </Text>
            </div>
            <Text font="secondary-body" as="p">
              {item.reason}
            </Text>
            {historical && (
              <Text font="secondary-body" as="p">
                This archived PDF does not replace the current legislation text.
              </Text>
            )}
            {!historical && !current && (
              <Text font="secondary-body" as="p">
                This PDF belongs to an earlier reviewed version. Equivalence to
                the current text has not been reconfirmed.
              </Text>
            )}
            <div>
              <Button
                size="sm"
                prominence="secondary"
                href={retainedPdfEvidenceUrl(
                  fileId,
                  item.id,
                  data.document_set_id
                )}
                target="_blank"
              >
                Open source PDF
              </Button>
            </div>
          </div>
        );
      })}
    </section>
  );
}
