"use client";

import type { ReactNode } from "react";
import { Text } from "@opal/components";
import { ChunkContent } from "@/sections/cards/ChunkChangeCard";

const chunkFieldOrder = [
  "id",
  "user_file_id",
  "position",
  "text",
  "chunk_type",
  "heading_path",
  "metadata",
  "validity_start_date",
  "validity_end_date",
  "status",
  "source",
  "supersedes_chunk_id",
  "superseded_by_chunk_id",
  "created_at",
  "updated_at",
] as const;

const alwaysVisibleFields = new Set([
  "validity_start_date",
  "validity_end_date",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && !Array.isArray(value) && typeof value === "object";
}

export function ReadOnlyFieldValue({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === "") {
    return <span className="text-text-03">—</span>;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-text-03">Empty</span>;
    }
    return (
      <div className="flex flex-col gap-1">
        {value.map((item, index) => (
          <div key={index} className="break-words text-sm text-text-05">
            {String(item)}
          </div>
        ))}
      </div>
    );
  }
  if (isRecord(value)) {
    const visibleEntries = Object.entries(value).filter(
      ([, child]) => child !== null && child !== undefined
    );
    if (visibleEntries.length === 0) {
      return <span className="text-text-03">Empty</span>;
    }
    return (
      <div className="flex flex-col gap-1.5">
        {visibleEntries.map(([key, child]) => (
          <div
            key={key}
            className="grid grid-cols-[minmax(6rem,0.35fr)_minmax(0,1fr)] gap-2"
          >
            <span className="break-words font-mono text-xs text-text-03">
              {key}
            </span>
            <ReadOnlyFieldValue value={child} />
          </div>
        ))}
      </div>
    );
  }
  return (
    <span className="whitespace-pre-wrap break-words text-sm text-text-05">
      {String(value)}
    </span>
  );
}

function FieldTable({
  title,
  fields,
  renderValue,
}: {
  title: string;
  fields: Record<string, unknown> | null;
  renderValue?: (key: string, value: unknown) => ReactNode;
}) {
  const visibleKeys = fields
    ? chunkFieldOrder.filter(
        (key) =>
          alwaysVisibleFields.has(key) ||
          (fields[key] !== null && fields[key] !== undefined)
      )
    : [];

  return (
    <div className="min-w-0 flex-1 overflow-hidden rounded-12 border border-border-02 bg-background-neutral-00">
      <div className="border-b border-border-02 bg-background-tint-01 px-3 py-2.5">
        <Text font="main-ui-action" color="text-04">
          {title}
        </Text>
      </div>
      {fields === null ? (
        <div className="p-3">
          <Text font="main-ui-body" color="text-03">
            (new — no existing chunk)
          </Text>
        </div>
      ) : (
        <div role="table" aria-label={`${title} chunk fields`}>
          {visibleKeys.map((key) => (
            <div
              key={key}
              role="row"
              className="grid grid-cols-1 gap-2 border-b border-border-01 px-3 py-2.5 last:border-b-0 sm:grid-cols-[minmax(9rem,0.35fr)_minmax(0,1fr)]"
            >
              <div role="rowheader">
                <span className="break-words font-mono text-xs text-text-03">
                  {key}
                </span>
              </div>
              <div role="cell" className="min-w-0">
                {renderValue ? (
                  renderValue(key, fields[key])
                ) : key === "text" && typeof fields[key] === "string" ? (
                  <ChunkContent text={fields[key]} />
                ) : (
                  <ReadOnlyFieldValue value={fields[key]} />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface ChunkFieldsReviewProps {
  before: Record<string, unknown> | null;
  after: Record<string, unknown>;
  renderAfterValue: (key: string, value: unknown) => ReactNode;
}

export default function ChunkFieldsReview({
  before,
  after,
  renderAfterValue,
}: ChunkFieldsReviewProps) {
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <FieldTable title="Before" fields={before} />
      <FieldTable title="After" fields={after} renderValue={renderAfterValue} />
    </div>
  );
}
