"use client";

import useSWR from "swr";
import { Text } from "@opal/components";
import { getAmendmentRuntime } from "@/lib/regulatory/amendments";

interface AmendmentRuntimePanelProps {
  batchId: number;
  status: string;
}

function gib(value: number | null): string {
  return value === null ? "Unknown" : `${(value / 1024 ** 3).toFixed(2)} GiB`;
}

function fresh(timestamp: string | null): boolean {
  if (!timestamp) return false;
  const age = Date.now() - Date.parse(timestamp);
  return age >= -5000 && age <= 20000;
}

export default function AmendmentRuntimePanel({
  batchId,
  status,
}: AmendmentRuntimePanelProps) {
  const running = status === "queued" || status === "analyzing";
  const { data, error } = useSWR(
    ["amendment-runtime", batchId],
    () => getAmendmentRuntime(batchId),
    {
      refreshInterval: () => (running ? 5000 : 0),
      keepPreviousData: false,
      revalidateOnFocus: running,
      shouldRetryOnError: running,
    }
  );
  const memoryFresh = !error && !!data && fresh(data.memory_checked_at);
  const activityFresh = !error && !!data && fresh(data.activity_checked_at);
  const memoryLabel =
    running && memoryFresh ? "Worker memory" : "Last worker memory";
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border-02 p-3">
      <Text font="main-ui-action">Analysis resources</Text>
      {!data || error ? (
        <Text font="secondary-body" color="text-03">
          Current resource measurements are unavailable.
        </Text>
      ) : (
        <>
          {data.analysis_model && (
            <Text font="secondary-body">
              {`Analysis model: ${data.analysis_model === "gemini-3.5-flash-lite" ? "Gemini 3.5 Flash Lite" : "Gemini 3.8 Flash"} · Vertex AI`}
            </Text>
          )}
          <Text font="secondary-body">
            {`${memoryLabel}: ${gib(data.current_bytes)} / ${gib(data.limit_bytes)} · Peak: ${gib(data.peak_bytes)} · Reserve: ${gib(data.reserve_bytes)}`}
          </Text>
          {running && !memoryFresh && (
            <Text font="secondary-body" color="text-03">
              Memory measurement is not current.
            </Text>
          )}
          <Text font="secondary-body">
            {status === "queued" && data.active === null
              ? "Queued: waiting for a worker to report task activity."
              : `${running && activityFresh ? "Active tasks" : "Last task count"}: ${data.active ?? "Unknown"} · Observed concurrency limit: ${data.max_parallel ?? "Unknown"}`}
          </Text>
          {status === "analyzing" && !activityFresh && (
            <Text font="secondary-body" color="text-03">
              Task count is not current.
            </Text>
          )}
          {running && activityFresh && data.calibrating && (
            <Text font="secondary-body" color="text-03">
              Measuring the initial tasks before expanding concurrency.
            </Text>
          )}
          {running && activityFresh && data.admission_limited && (
            <Text font="secondary-body" color="text-03">
              Waiting for memory capacity to start another task.
            </Text>
          )}
          {running && activityFresh && data.dependency_limited && (
            <Text font="secondary-body" color="text-03">
              Changes to the same provision are waiting for the preceding
              change.
            </Text>
          )}
          <Text font="secondary-body" color="text-03">
            {`Text submitted for analysis: ${data.raw_text_chars.toLocaleString()} characters. Memory covers the worker container, including other work in that container.`}
          </Text>
          <Text font="secondary-body" color="text-03">
            {`Memory measured: ${data.memory_checked_at ? new Date(data.memory_checked_at).toLocaleTimeString() : "Unknown"} · Tasks measured: ${data.activity_checked_at ? new Date(data.activity_checked_at).toLocaleTimeString() : "Unknown"}`}
          </Text>
        </>
      )}
    </div>
  );
}
