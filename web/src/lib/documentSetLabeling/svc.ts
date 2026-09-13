import type {
  LabelingRun,
  LabelingRunItemsPage,
  LabelingSetup,
  LabelingTaxonomySummary,
  LabelSettingsSnapshot,
  LabelSettingsUpdate,
  TaxonomyInput,
} from "@/lib/documentSetLabeling/interfaces";

export class LabelingApiError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = "LabelingApiError";
  }
}

export function labelingBaseUrl(documentSetId: number): string {
  return `/api/manage/admin/document-set/${documentSetId}/labeling`;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = init ? await fetch(url, init) : await fetch(url);
  if (!response.ok) {
    let detail: unknown;
    try {
      detail = (await response.json())?.detail;
    } catch {
      detail = undefined;
    }
    throw new LabelingApiError(
      typeof detail === "string"
        ? detail
        : `Request failed (${response.status})`,
      response.status
    );
  }
  return response.json();
}

function postJson<T>(url: string, body?: unknown): Promise<T> {
  return requestJson<T>(url, {
    method: "POST",
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function putJson<T>(url: string, body: unknown): Promise<T> {
  return requestJson<T>(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getLabelingSetup(
  documentSetId: number
): Promise<LabelingSetup> {
  return requestJson(`${labelingBaseUrl(documentSetId)}/setup`);
}

export function listLabelingRuns(
  documentSetId: number
): Promise<LabelingRun[]> {
  return requestJson(`${labelingBaseUrl(documentSetId)}/runs`);
}

export function getLabelingRun(
  documentSetId: number,
  runId: string
): Promise<LabelingRun> {
  return requestJson(`${labelingBaseUrl(documentSetId)}/runs/${runId}`);
}

export function getLabelingRunItems(
  documentSetId: number,
  runId: string,
  offset: number,
  limit: number
): Promise<LabelingRunItemsPage> {
  return requestJson(
    `${labelingBaseUrl(documentSetId)}/runs/${runId}/items?offset=${offset}&limit=${limit}`
  );
}

export function createLabelingTaxonomy(
  documentSetId: number,
  taxonomy: TaxonomyInput
): Promise<LabelingTaxonomySummary> {
  return postJson(`${labelingBaseUrl(documentSetId)}/taxonomies`, taxonomy);
}

export function getLabelSettings(
  documentSetId: number
): Promise<LabelSettingsSnapshot> {
  return requestJson(`${labelingBaseUrl(documentSetId)}/label-settings`);
}

export function updateLabelSettings(
  documentSetId: number,
  update: LabelSettingsUpdate
): Promise<LabelSettingsSnapshot> {
  return putJson(`${labelingBaseUrl(documentSetId)}/label-settings`, update);
}

export function startLabelingRun(
  documentSetId: number,
  body: {
    model_configuration_id: number;
    idempotency_key: string;
  }
): Promise<LabelingRun> {
  return postJson(`${labelingBaseUrl(documentSetId)}/runs`, body);
}

export function cancelLabelingRun(
  documentSetId: number,
  runId: string
): Promise<LabelingRun> {
  return postJson(`${labelingBaseUrl(documentSetId)}/runs/${runId}/cancel`);
}

export function retryLabelingRun(
  documentSetId: number,
  runId: string
): Promise<LabelingRun> {
  return postJson(`${labelingBaseUrl(documentSetId)}/runs/${runId}/retry`);
}
