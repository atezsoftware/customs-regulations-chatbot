import useSWR from "swr";

import type { LabelingRun } from "@/lib/documentSetLabeling/interfaces";
import {
  getLabelingRun,
  getLabelingRunItems,
  getLabelingSetup,
  getLabelSettings,
  labelingBaseUrl,
  listLabelingRuns,
} from "@/lib/documentSetLabeling/svc";

export function useLabelSettings(documentSetId: number) {
  return useSWR(
    `${labelingBaseUrl(documentSetId)}/label-settings`,
    () => getLabelSettings(documentSetId),
    { shouldRetryOnError: false }
  );
}

export function isActiveLabelingRun(run: LabelingRun): boolean {
  return run.status === "queued" || run.status === "running";
}

export function useLabelingSetup(documentSetId: number) {
  return useSWR(
    `${labelingBaseUrl(documentSetId)}/setup`,
    () => getLabelingSetup(documentSetId),
    { shouldRetryOnError: false }
  );
}

export function useLabelingRuns(documentSetId: number) {
  return useSWR(
    `${labelingBaseUrl(documentSetId)}/runs`,
    () => listLabelingRuns(documentSetId),
    {
      refreshInterval: (runs) => (runs?.some(isActiveLabelingRun) ? 5000 : 0),
      keepPreviousData: true,
    }
  );
}

export function useLabelingRun(documentSetId: number, runId: string | null) {
  return useSWR(
    runId ? `${labelingBaseUrl(documentSetId)}/runs/${runId}` : null,
    () => getLabelingRun(documentSetId, runId!),
    {
      refreshInterval: (run) => (run && isActiveLabelingRun(run) ? 5000 : 0),
      keepPreviousData: true,
    }
  );
}

export function useLabelingRunItems(
  documentSetId: number,
  runId: string | null,
  offset: number,
  limit: number,
  shouldPoll: boolean
) {
  return useSWR(
    runId
      ? `${labelingBaseUrl(documentSetId)}/runs/${runId}/items?offset=${offset}&limit=${limit}`
      : null,
    () => getLabelingRunItems(documentSetId, runId!, offset, limit),
    { refreshInterval: shouldPoll ? 5000 : 0, keepPreviousData: true }
  );
}
