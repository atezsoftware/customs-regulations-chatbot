"use client";

import { useEffect, useMemo, useState } from "react";

import {
  Button,
  Card,
  InputTypeIn,
  MessageCard,
  ProgressBar,
  Tag,
  Text,
} from "@opal/components";
import { Content, PageLoader } from "@opal/layouts";
import {
  SvgPlayCircle,
  SvgRefreshCw,
  SvgSettings,
  SvgStopCircle,
  SvgTag,
} from "@opal/icons";

import LabelSettingsModal from "@/app/admin/documents/sets/[documentSetId]/labeling/LabelSettingsModal";
import InputSelect from "@/refresh-components/inputs/InputSelect";
import type {
  LabelingRun,
  LabelingRunItemsPage,
  LabelingRunStatus,
} from "@/lib/documentSetLabeling/interfaces";
import {
  isActiveLabelingRun,
  useLabelingRun,
  useLabelingRunItems,
  useLabelingRuns,
  useLabelingSetup,
} from "@/lib/documentSetLabeling/hooks";
import {
  cancelLabelingRun,
  LabelingApiError,
  retryLabelingRun,
  startLabelingRun,
} from "@/lib/documentSetLabeling/svc";

const ITEM_PAGE_SIZE = 25;

interface LabelingWorkspaceProps {
  documentSetId: number;
}

interface PendingStart {
  key: string;
  providerId: number;
  usesBundledTaxonomy: true;
}

function displayName(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function statusColor(status: LabelingRunStatus) {
  if (status === "completed") return "green" as const;
  if (status === "failed" || status === "cancelled") return "red" as const;
  if (status === "completed_with_errors") return "amber" as const;
  return "blue" as const;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}

function startStorageKey(documentSetId: number): string {
  return `document-set-labeling:${documentSetId}:pending-start`;
}

function getPendingStart(
  documentSetId: number,
  providerId: number
): PendingStart {
  const storageKey = startStorageKey(documentSetId);
  const stored = window.localStorage.getItem(storageKey);
  if (stored) {
    try {
      const pending = JSON.parse(stored) as PendingStart;
      if (
        typeof pending.key === "string" &&
        pending.providerId === providerId &&
        pending.usesBundledTaxonomy === true
      )
        return pending;
    } catch {
      window.localStorage.removeItem(storageKey);
    }
  }
  const pending = {
    key: window.crypto.randomUUID(),
    providerId,
    usesBundledTaxonomy: true as const,
  };
  window.localStorage.setItem(storageKey, JSON.stringify(pending));
  return pending;
}

export default function LabelingWorkspace({
  documentSetId,
}: LabelingWorkspaceProps) {
  const {
    data: setup,
    error: setupError,
    isLoading: setupLoading,
    mutate: mutateSetup,
  } = useLabelingSetup(documentSetId);
  const {
    data: runs = [],
    error: runsError,
    isLoading: runsLoading,
    mutate: mutateRuns,
  } = useLabelingRuns(documentSetId);
  const [providerId, setProviderId] = useState("");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [itemOffset, setItemOffset] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const hasActiveRun =
    runs.some(isActiveLabelingRun) ||
    Boolean(
      setup?.active_run_id &&
      !runs.some((run) => run.id === setup.active_run_id)
    );

  useEffect(() => {
    if (!providerId && setup?.providers.length === 1)
      setProviderId(String(setup.providers[0]!.id));
  }, [providerId, setup]);

  useEffect(() => {
    if (selectedRunId || runs.length === 0) return;
    setSelectedRunId(
      setup?.active_run_id && runs.some((run) => run.id === setup.active_run_id)
        ? setup.active_run_id
        : runs[0]!.id
    );
  }, [runs, selectedRunId, setup?.active_run_id]);

  useEffect(() => {
    if (hasActiveRun) {
      window.localStorage.removeItem(startStorageKey(documentSetId));
    }
  }, [documentSetId, hasActiveRun]);

  const selectedSummary = runs.find((run) => run.id === selectedRunId) ?? null;
  const { data: selectedDetail } = useLabelingRun(documentSetId, selectedRunId);
  const selectedRun =
    selectedDetail?.id === selectedRunId ? selectedDetail : selectedSummary;
  const selectedRunIsActive = selectedRun
    ? isActiveLabelingRun(selectedRun)
    : false;
  const { data: itemPage, error: itemError } = useLabelingRunItems(
    documentSetId,
    selectedRunId,
    itemOffset,
    ITEM_PAGE_SIZE,
    selectedRunIsActive
  );

  const readiness = useMemo(
    () => ({
      provider: Boolean(providerId),
      chunks: Boolean(setup?.counts.canonical_chunks),
      idle: !runsError && !hasActiveRun,
      configuration: !setup?.configuration_errors?.length,
    }),
    [
      providerId,
      hasActiveRun,
      runsError,
      setup?.counts.canonical_chunks,
      setup?.configuration_errors,
    ]
  );
  const canStart = Object.values(readiness).every(Boolean) && !isSubmitting;

  const accessError = [setupError, runsError].find(
    (error) =>
      error instanceof LabelingApiError &&
      (error.status === 403 || error.status === 404)
  );
  if (accessError) {
    return (
      <MessageCard
        variant="warning"
        title="Chunk labeling is unavailable"
        description="This document set cannot be labeled or you do not have access to it."
      />
    );
  }
  if ((setupLoading || runsLoading) && !setup) return <PageLoader />;
  if (!setup) {
    return (
      <MessageCard
        variant="error"
        title="Could not load chunk labeling"
        description={errorMessage(setupError ?? runsError)}
      />
    );
  }

  async function handleStart() {
    if (!canStart) return;
    setIsSubmitting(true);
    setActionError(null);
    setNotice(null);
    const numericProviderId = Number(providerId);
    const pending = getPendingStart(documentSetId, numericProviderId);
    try {
      const run = await startLabelingRun(documentSetId, {
        model_configuration_id: numericProviderId,
        idempotency_key: pending.key,
      });
      window.localStorage.removeItem(startStorageKey(documentSetId));
      setSelectedRunId(run.id);
      setItemOffset(0);
      setNotice("Labeling run queued");
      void mutateRuns();
      void mutateSetup();
    } catch (error) {
      const requestWasRejected =
        error instanceof LabelingApiError &&
        error.status >= 400 &&
        error.status < 500 &&
        error.status !== 408;
      if (requestWasRejected)
        window.localStorage.removeItem(startStorageKey(documentSetId));
      setActionError(
        requestWasRejected
          ? error.message
          : "The start request could not be confirmed. Retry to safely check the same request."
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleRunAction(action: "cancel" | "retry", run: LabelingRun) {
    setIsSubmitting(true);
    setActionError(null);
    try {
      const updated =
        action === "cancel"
          ? await cancelLabelingRun(documentSetId, run.id)
          : await retryLabelingRun(documentSetId, run.id);
      setSelectedRunId(updated.id);
      setItemOffset(0);
      setNotice(
        action === "cancel" ? "Cancellation requested" : "New full run queued"
      );
      void mutateRuns();
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <MessageCard
        variant="info"
        title="Batch labeling runs asynchronously"
        description="Gemini Batch can take several hours. Labels apply to existing canonical chunks and propagate to their derived chunks; generated context is not a separate labeled chunk."
      />
      {setup.warnings.map((warning) => (
        <MessageCard
          key={warning}
          variant="warning"
          title={warning}
          titleMaxLines={undefined}
        />
      ))}
      {setup.configuration_errors?.map((configurationError) => (
        <MessageCard
          key={configurationError}
          variant="error"
          title={configurationError}
          titleMaxLines={undefined}
        />
      ))}
      {actionError && (
        <MessageCard
          variant="error"
          title={actionError}
          titleMaxLines={undefined}
        />
      )}
      {runsError && !accessError && (
        <MessageCard
          variant="error"
          title="Could not load run history"
          description={errorMessage(runsError)}
        />
      )}
      {notice && <MessageCard variant="success" title={notice} />}

      <Card border="solid" padding="md">
        <div className="flex flex-col gap-4">
          <Content
            sizePreset="section"
            variant="section"
            icon={SvgTag}
            title="Labeling setup"
            description="Label names and descriptions are sent to Gemini automatically."
          />
          <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
            <ReadinessItem
              ready={setup.default_label_count > 0}
              readyText={`${setup.default_label_count} labels ready`}
              blockedText="Label definitions unavailable"
            />
            <ReadinessItem
              ready={readiness.provider}
              readyText={`${setup.providers.length} Gemini connection${setup.providers.length === 1 ? "" : "s"}`}
              blockedText={
                setup.providers.length
                  ? "Select a Gemini connection"
                  : "No Gemini connection configured"
              }
            />
            <ReadinessItem
              ready={readiness.chunks}
              readyText={`${setup.counts.canonical_chunks} canonical chunks ready`}
              blockedText="No canonical chunks ready"
            />
          </div>
          <Text font="secondary-body" color="text-03">
            {`${setup.counts.files} files · ${setup.counts.canonical_chunks} canonical chunks · ${setup.counts.derived_chunks} derived chunks`}
          </Text>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Field label="Gemini connection">
              <InputSelect
                value={providerId}
                onValueChange={setProviderId}
                disabled={!setup.providers.length}
              >
                <InputSelect.Trigger
                  aria-label="Gemini connection"
                  placeholder={
                    setup.providers.length
                      ? "Select a Gemini connection"
                      : "Configure a Gemini connection first"
                  }
                />
                <InputSelect.Content>
                  {setup.providers.map((provider) => (
                    <InputSelect.Item
                      key={provider.id}
                      value={String(provider.id)}
                    >
                      {provider.name}
                    </InputSelect.Item>
                  ))}
                </InputSelect.Content>
              </InputSelect>
            </Field>
            <Field label="Model">
              <InputTypeIn
                aria-label="Model"
                value={setup.model}
                variant="readOnly"
              />
            </Field>
          </div>

          <div className="flex justify-end gap-2">
            <Button
              icon={SvgSettings}
              prominence="secondary"
              onClick={() => setSettingsOpen(true)}
            >
              Label Settings
            </Button>
            <Button
              icon={SvgPlayCircle}
              prominence="primary"
              disabled={!canStart}
              tooltip={
                !readiness.idle
                  ? "A labeling run is already active."
                  : undefined
              }
              onClick={() => void handleStart()}
            >
              Start Labeling
            </Button>
          </div>
        </div>
      </Card>

      {settingsOpen && (
        <LabelSettingsModal
          documentSetId={documentSetId}
          onClose={() => setSettingsOpen(false)}
          onSaved={() => void mutateSetup()}
        />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <Card border="solid" padding="md">
          <div className="flex flex-col gap-2">
            <Text as="h2" font="heading-h3" color="text-04">
              Run history
            </Text>
            {runs.length === 0 ? (
              <Text font="main-ui-body" color="text-03">
                No labeling runs yet.
              </Text>
            ) : (
              runs.map((run) => (
                <Button
                  key={run.id}
                  prominence={
                    run.id === selectedRunId ? "secondary" : "tertiary"
                  }
                  size="fit"
                  onClick={() => {
                    setSelectedRunId(run.id);
                    setItemOffset(0);
                  }}
                >
                  {`${run.taxonomy_name} · ${displayName(run.status)}`}
                </Button>
              ))
            )}
          </div>
        </Card>

        <RunDetails
          run={selectedRun}
          itemPage={itemPage}
          itemError={itemError}
          itemOffset={itemOffset}
          busy={isSubmitting}
          onCancel={(run) => void handleRunAction("cancel", run)}
          onRetry={(run) => void handleRunAction("retry", run)}
          onPageChange={setItemOffset}
        />
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Text font="secondary-action" color="text-03">
        {label}
      </Text>
      {children}
    </div>
  );
}

function ReadinessItem({
  ready,
  readyText,
  blockedText,
}: {
  ready: boolean;
  readyText: string;
  blockedText: string;
}) {
  return (
    <Card padding="sm" background="heavy">
      <Text font="main-ui-body" color={ready ? "status-success-05" : "text-03"}>
        {ready ? readyText : blockedText}
      </Text>
    </Card>
  );
}

interface RunDetailsProps {
  run: LabelingRun | null;
  itemPage: LabelingRunItemsPage | undefined;
  itemError: unknown;
  itemOffset: number;
  busy: boolean;
  onCancel: (run: LabelingRun) => void;
  onRetry: (run: LabelingRun) => void;
  onPageChange: (offset: number) => void;
}

function RunDetails({
  run,
  itemPage,
  itemError,
  itemOffset,
  busy,
  onCancel,
  onRetry,
  onPageChange,
}: RunDetailsProps) {
  if (!run) {
    return (
      <MessageCard
        title="Select a run"
        description="Run status and chunk outcomes appear here."
      />
    );
  }
  const processed = Math.min(
    run.total_chunks,
    run.completed_chunks + run.failed_chunks + run.stale_chunks
  );
  const canRetry = ["failed", "cancelled", "completed_with_errors"].includes(
    run.status
  );
  return (
    <Card border="solid" padding="md">
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="flex flex-col gap-1">
            <Text as="h2" font="heading-h3" color="text-04">
              {run.taxonomy_name}
            </Text>
            <div className="flex flex-wrap gap-1">
              <Tag
                title={displayName(run.status)}
                color={statusColor(run.status)}
              />
              <Tag title={displayName(run.stage)} color="gray" />
              <Tag title={run.model} color="purple" />
            </div>
          </div>
          <div className="flex gap-2">
            {isActiveLabelingRun(run) && (
              <Button
                icon={SvgStopCircle}
                variant="danger"
                prominence="secondary"
                disabled={busy || run.cancel_requested}
                onClick={() => onCancel(run)}
              >
                {run.cancel_requested ? "Cancellation requested" : "Cancel run"}
              </Button>
            )}
            {canRetry && (
              <Button
                icon={SvgRefreshCw}
                prominence="secondary"
                disabled={busy}
                tooltip="Creates a new full run from a fresh snapshot."
                onClick={() => onRetry(run)}
              >
                Retry full run
              </Button>
            )}
          </div>
        </div>

        {run.error && (
          <MessageCard
            variant="error"
            title={run.error}
            titleMaxLines={undefined}
          />
        )}
        {run.stage === "waiting" ? (
          <MessageCard
            variant="pending"
            title="Waiting for Gemini Batch"
            description="Results and counts update when the provider returns them; this stage can take several hours."
          />
        ) : (
          <ProgressBar
            value={processed}
            max={run.total_chunks}
            aria-label="Processed canonical chunks"
          />
        )}
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
          <Count label="Total" value={run.total_chunks} />
          <Count label="Completed" value={run.completed_chunks} />
          <Count label="Failed" value={run.failed_chunks} />
          <Count label="Stale" value={run.stale_chunks} />
          <Count label="Derived" value={run.derived_chunks} />
          <Count
            label="Unresolved derived"
            value={run.unresolved_derived_chunks}
          />
        </div>

        <Text as="h3" font="heading-h3" color="text-04">
          Chunk outcomes
        </Text>
        {itemError ? (
          <MessageCard
            variant="error"
            title="Could not load chunk outcomes"
            description={errorMessage(itemError)}
          />
        ) : itemPage?.items.length ? (
          <div className="flex flex-col gap-2">
            {itemPage.items.map((item) => (
              <Card key={item.chunk_id} padding="sm" background="heavy">
                <div className="flex flex-col gap-1">
                  <div className="flex flex-wrap items-center gap-1">
                    <Text font="main-ui-action" color="text-04">
                      {item.chunk_id}
                    </Text>
                    <Tag
                      title={displayName(item.status)}
                      color={item.error ? "red" : "gray"}
                    />
                    {item.labels.map((label) => (
                      <Tag key={label} title={label} color="blue" />
                    ))}
                  </div>
                  <Text
                    font="secondary-body"
                    color="text-03"
                  >{`File ${item.file_id}`}</Text>
                  {item.error && (
                    <Text font="secondary-body" color="status-error-05">
                      {item.error}
                    </Text>
                  )}
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <Text font="main-ui-body" color="text-03">
            No chunk outcomes are available yet.
          </Text>
        )}
        {itemPage && itemPage.total > ITEM_PAGE_SIZE && (
          <div className="flex justify-between">
            <Button
              prominence="tertiary"
              disabled={itemOffset === 0}
              onClick={() =>
                onPageChange(Math.max(0, itemOffset - ITEM_PAGE_SIZE))
              }
            >
              Previous
            </Button>
            <Text font="secondary-body" color="text-03">
              {`${itemOffset + 1}–${Math.min(itemOffset + ITEM_PAGE_SIZE, itemPage.total)} of ${itemPage.total}`}
            </Text>
            <Button
              prominence="tertiary"
              disabled={itemOffset + ITEM_PAGE_SIZE >= itemPage.total}
              onClick={() => onPageChange(itemOffset + ITEM_PAGE_SIZE)}
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}

function Count({ label, value }: { label: string; value: number }) {
  return (
    <Card padding="sm" background="heavy">
      <div className="flex flex-col gap-0.5">
        <Text font="figure-small-value" color="text-03">
          {label}
        </Text>
        <Text font="heading-h3" color="text-04">
          {value.toLocaleString()}
        </Text>
      </div>
    </Card>
  );
}
