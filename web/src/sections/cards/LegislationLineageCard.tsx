"use client";

import useSWR from "swr";
import { Button, Tag, Text } from "@opal/components";
import { getLegislationLineage } from "@/lib/regulatory/documentIntake";

interface LegislationLineageCardProps {
  fileId: string;
  onOpenRelatedFile?: (fileId: string) => void;
}

export default function LegislationLineageCard({
  fileId,
  onOpenRelatedFile,
}: LegislationLineageCardProps) {
  const { data, error } = useSWR(["legislation-lineage", fileId], () =>
    getLegislationLineage(fileId)
  );
  return (
    <section
      aria-label="Legislation history"
      className="flex flex-col gap-2 rounded-12 border border-border-02 p-3"
    >
      <Text as="h3" font="main-ui-action">
        Legislation history
      </Text>
      {error && (
        <Text color="status-error-05">
          {error instanceof Error
            ? error.message
            : "Could not load legislation history."}
        </Text>
      )}
      {!data && !error && (
        <Text font="secondary-body">Loading legislation history…</Text>
      )}
      {data && (
        <>
          <Tag
            title={
              data.status === "repealed"
                ? "Repealed"
                : data.status === "repeal_scheduled"
                  ? "Repeal scheduled"
                  : "Current"
            }
          />
          {data.effective_date && (
            <Text font="secondary-body">{`Effective date: ${data.effective_date}`}</Text>
          )}
          {data.predecessors.length === 0 && data.successors.length === 0 && (
            <Text font="secondary-body" color="text-03">
              No predecessor or successor relationship is recorded.
            </Text>
          )}
          {[
            ...data.predecessors.map((edge) => ({
              ...edge,
              direction: "Predecessor",
            })),
            ...data.successors.map((edge) => ({
              ...edge,
              direction: "Successor",
            })),
          ].map((edge) => (
            <div
              key={`${edge.direction}:${edge.user_file_id}`}
              className="flex flex-col gap-1 border-t border-border-02 pt-2"
            >
              <Text font="secondary-action">{`${edge.direction}: ${edge.title}`}</Text>
              <Text
                font="secondary-body"
                color="text-03"
              >{`${edge.relation === "repeals" ? (edge.direction === "Successor" ? "This legislation is repealed by this successor." : "This document repeals the predecessor.") : "Legislation succession"}${edge.effective_date ? ` Effective: ${edge.effective_date}` : ""}`}</Text>
              {edge.evidence && (
                <Text as="p" font="secondary-body">
                  {edge.evidence}
                </Text>
              )}
              {onOpenRelatedFile && (
                <div>
                  <Button
                    prominence="tertiary"
                    onClick={() => onOpenRelatedFile(edge.user_file_id)}
                  >{`Open ${edge.title}`}</Button>
                </div>
              )}
            </div>
          ))}
        </>
      )}
    </section>
  );
}
