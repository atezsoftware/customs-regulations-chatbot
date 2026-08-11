import type { ReactNode } from "react";

import { Card, Tag, Text } from "@opal/components";
import type { RichStr } from "@opal/types";

import type {
  BenchmarkModelSelection,
  BenchmarkRun,
} from "@/lib/regulatory/benchmark";

const STATUS_PRESENTATION: Record<
  BenchmarkRun["status"],
  { label: string; color: "green" | "blue" | "red" | "amber" | "gray" }
> = {
  pending: { label: "Pending", color: "amber" },
  queued: { label: "Queued", color: "amber" },
  running: { label: "Running", color: "blue" },
  completed: { label: "Completed", color: "green" },
  error: { label: "Failed", color: "red" },
  cancelled: { label: "Cancelled", color: "gray" },
};

interface FormFieldProps {
  label: string | RichStr;
  hint?: string | RichStr;
  children: ReactNode;
}

export function FormField({ label, hint, children }: FormFieldProps) {
  return (
    <label className="flex flex-col gap-1.5">
      <Text font="main-ui-action" color="text-04">
        {label}
      </Text>
      {hint && (
        <Text font="secondary-body" color="text-02">
          {hint}
        </Text>
      )}
      {children}
    </label>
  );
}

interface StatusBadgeProps {
  status: string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const presentation =
    status in STATUS_PRESENTATION
      ? STATUS_PRESENTATION[status as BenchmarkRun["status"]]
      : { label: status, color: "gray" as const };
  return <Tag title={presentation.label} color={presentation.color} />;
}

interface MetricProps {
  label: string | RichStr;
  value: string | RichStr;
}

export function Metric({ label, value }: MetricProps) {
  return (
    <Card border="solid" padding="sm">
      <div className="flex flex-col gap-1">
        <Text font="secondary-body" color="text-02">
          {label}
        </Text>
        <Text font="main-ui-action" color="text-05">
          {value}
        </Text>
      </div>
    </Card>
  );
}

export const modelKey = (
  model: Pick<BenchmarkModelSelection, "provider" | "provider_id" | "model_id">
) => `${model.provider_id ?? "legacy"}::${model.provider}::${model.model_id}`;

export const modelIdentityLabel = (
  model: Pick<BenchmarkModelSelection, "provider" | "provider_id" | "model_id">
) => `${model.provider} #${model.provider_id ?? "legacy"} · ${model.model_id}`;

export const formatPercent = (value: number | null) =>
  value == null ? "—" : `${Math.round(value * 100)}%`;

export const formatCost = (value: number | null) => {
  if (value == null) return "Unavailable";
  const dollars = value / 100;
  return dollars > 0 && dollars < 0.0001
    ? "<$0.0001"
    : `$${dollars.toFixed(4)}`;
};

export const formatDuration = (value: number | null) =>
  value == null ? "—" : `${(value / 1000).toFixed(1)}s`;
