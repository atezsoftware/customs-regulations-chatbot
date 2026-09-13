"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  Button,
  InputTextArea,
  InputTypeIn,
  MessageCard,
  Modal,
  Text,
} from "@opal/components";
import { PageLoader } from "@opal/layouts";
import { SvgPlus, SvgSettings, SvgTrash } from "@opal/icons";

import type {
  LabelSettingsSnapshot,
  TaxonomyLabelInput,
} from "@/lib/documentSetLabeling/interfaces";
import { useLabelSettings } from "@/lib/documentSetLabeling/hooks";
import { updateLabelSettings } from "@/lib/documentSetLabeling/svc";

const MAX_LABELS = 1024;
const MAX_LABEL_ID_LENGTH = 100;
const MAX_NAME_LENGTH = 200;
const MAX_DESCRIPTION_LENGTH = 8000;
const LABEL_ID_PATTERN = new RegExp("^[\\p{L}\\p{N}_.:-]+$", "u");

interface DraftLabel extends TaxonomyLabelInput {
  key: string;
  isNew: boolean;
}

interface LabelSettingsModalProps {
  documentSetId: number;
  onClose: () => void;
  onSaved: (snapshot: LabelSettingsSnapshot) => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}

function toDraftLabels(labels: TaxonomyLabelInput[]): DraftLabel[] {
  return labels.map((label) => ({
    ...label,
    key: `existing:${label.id}`,
    isNew: false,
  }));
}

function normalizedLabels(draft: DraftLabel[]): TaxonomyLabelInput[] {
  return draft.map(({ id, name, description }) => ({
    id: id.trim(),
    name: name.trim(),
    description: description.trim(),
  }));
}

function validateLabels(labels: TaxonomyLabelInput[]): string | null {
  if (labels.length === 0) return "At least one label is required.";
  if (labels.length > MAX_LABELS)
    return `No more than ${MAX_LABELS} labels are allowed.`;

  const seenIds = new Set<string>();
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]!;
    const number = index + 1;
    if (label.id.length > MAX_LABEL_ID_LENGTH)
      return `Label ${number} IDs must be ${MAX_LABEL_ID_LENGTH} characters or fewer.`;
    if (!label.id || !LABEL_ID_PATTERN.test(label.id))
      return `Label ${number} IDs may contain only letters, numbers, underscores, periods, colons, and hyphens.`;
    if (seenIds.has(label.id)) return `Label ID ${label.id} must be unique.`;
    seenIds.add(label.id);
    if (!label.name || label.name.length > MAX_NAME_LENGTH)
      return `Label ${number} names must be between 1 and ${MAX_NAME_LENGTH} characters.`;
    if (!label.description || label.description.length > MAX_DESCRIPTION_LENGTH)
      return `Label ${number} descriptions must be between 1 and ${MAX_DESCRIPTION_LENGTH} characters.`;
  }
  return null;
}

export default function LabelSettingsModal({
  documentSetId,
  onClose,
  onSaved,
}: LabelSettingsModalProps) {
  const { data, error, isLoading, mutate } = useLabelSettings(documentSetId);
  const [draft, setDraft] = useState<DraftLabel[]>([]);
  const [revision, setRevision] = useState<number | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [reloading, setReloading] = useState(false);
  const busy = saving || reloading;

  const loadSnapshot = useCallback((snapshot: LabelSettingsSnapshot) => {
    const labels = toDraftLabels(snapshot.labels);
    setDraft(labels);
    setRevision(snapshot.revision);
    setSelectedKey(labels[0]?.key ?? null);
    setSearch("");
    setValidationError(null);
    setSaveError(null);
  }, []);

  useEffect(() => {
    if (data && revision === null) loadSnapshot(data);
  }, [data, loadSnapshot, revision]);

  const selected = draft.find((label) => label.key === selectedKey) ?? null;
  const visibleLabels = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    if (!query) return draft;
    return draft.filter((label) =>
      `${label.id} ${label.name} ${label.description}`
        .toLocaleLowerCase()
        .includes(query)
    );
  }, [draft, search]);

  function updateSelected(field: keyof TaxonomyLabelInput, value: string) {
    if (!selectedKey || busy) return;
    setDraft((labels) =>
      labels.map((label) =>
        label.key === selectedKey ? { ...label, [field]: value } : label
      )
    );
    setValidationError(null);
    setSaveError(null);
  }

  function addLabel() {
    if (busy) return;
    const label: DraftLabel = {
      key: `new:${window.crypto.randomUUID()}`,
      id: "",
      name: "",
      description: "",
      isNew: true,
    };
    setDraft((labels) => [...labels, label]);
    setSelectedKey(label.key);
    setSearch("");
    setValidationError(null);
    setSaveError(null);
  }

  function deleteSelected() {
    if (!selectedKey || draft.length <= 1 || busy) return;
    const nextDraft = draft.filter((label) => label.key !== selectedKey);
    setDraft(nextDraft);
    setSelectedKey(nextDraft[0]?.key ?? null);
    setSearch("");
    setValidationError(null);
    setSaveError(null);
  }

  async function save() {
    if (revision === null || busy) return;
    const labels = normalizedLabels(draft);
    const nextValidationError = validateLabels(labels);
    if (nextValidationError) {
      setValidationError(nextValidationError);
      return;
    }
    setSaving(true);
    setValidationError(null);
    setSaveError(null);
    try {
      const saved = await updateLabelSettings(documentSetId, {
        expected_revision: revision,
        labels,
      });
      await mutate(saved, false);
      onSaved(saved);
      onClose();
    } catch (requestError) {
      setSaveError(errorMessage(requestError));
    } finally {
      setSaving(false);
    }
  }

  async function reload() {
    if (busy) return;
    setReloading(true);
    try {
      const latest = await mutate();
      if (latest) loadSnapshot(latest);
    } catch (requestError) {
      setSaveError(errorMessage(requestError));
    } finally {
      setReloading(false);
    }
  }

  return (
    <Modal open onOpenChange={(open) => !open && !busy && onClose()}>
      <Modal.Content width="lg" height="lg">
        <Modal.Header
          icon={SvgSettings}
          title="Label Settings"
          description="Shared across all document sets in this tenant. Saved changes apply to future runs; in-flight runs keep their existing label snapshot."
          onClose={busy ? undefined : onClose}
        />
        <Modal.Body>
          {isLoading && !data ? (
            <PageLoader />
          ) : error && !data ? (
            <MessageCard
              variant="error"
              title="Could not load label settings"
              description={errorMessage(error)}
            />
          ) : (
            <div className="flex min-h-0 flex-col gap-3 md:flex-row">
              <div className="flex min-h-0 flex-1 flex-col gap-2 md:max-w-[18rem]">
                <InputTypeIn
                  aria-label="Search labels"
                  searchIcon
                  clearButton
                  placeholder="Search labels"
                  value={search}
                  variant={busy ? "disabled" : "primary"}
                  onChange={(event) => setSearch(event.target.value)}
                />
                <Button
                  type="button"
                  prominence="secondary"
                  icon={SvgPlus}
                  disabled={busy || draft.length >= MAX_LABELS}
                  onClick={addLabel}
                >
                  Add label
                </Button>
                <div className="flex min-h-0 flex-col gap-1 overflow-y-auto">
                  {visibleLabels.length ? (
                    visibleLabels.map((label) => (
                      <Button
                        key={label.key}
                        type="button"
                        size="fit"
                        prominence={
                          label.key === selectedKey ? "secondary" : "tertiary"
                        }
                        disabled={busy}
                        onClick={() => setSelectedKey(label.key)}
                      >
                        {`${label.id || "New label"} · ${label.name || "Untitled"}`}
                      </Button>
                    ))
                  ) : (
                    <Text font="main-ui-body" color="text-03">
                      No labels match this search.
                    </Text>
                  )}
                </div>
              </div>

              <div className="flex min-w-0 flex-[2] flex-col gap-3">
                {selected ? (
                  <>
                    <Field label="Label ID">
                      <InputTypeIn
                        aria-label="Label ID"
                        maxLength={MAX_LABEL_ID_LENGTH}
                        value={selected.id}
                        variant={
                          busy
                            ? "disabled"
                            : selected.isNew
                              ? "primary"
                              : "readOnly"
                        }
                        onChange={(event) =>
                          updateSelected("id", event.target.value)
                        }
                      />
                    </Field>
                    <Field label="Name">
                      <InputTypeIn
                        aria-label="Label name"
                        maxLength={MAX_NAME_LENGTH}
                        value={selected.name}
                        variant={busy ? "disabled" : "primary"}
                        onChange={(event) =>
                          updateSelected("name", event.target.value)
                        }
                      />
                    </Field>
                    <Field label="Description">
                      <InputTextArea
                        aria-label="Label description"
                        maxLength={MAX_DESCRIPTION_LENGTH}
                        rows={8}
                        maxRows={14}
                        autoResize
                        value={selected.description}
                        variant={busy ? "disabled" : "primary"}
                        onChange={(event) =>
                          updateSelected("description", event.target.value)
                        }
                      />
                    </Field>
                    <div className="flex justify-end">
                      <Button
                        type="button"
                        variant="danger"
                        prominence="tertiary"
                        icon={SvgTrash}
                        disabled={busy || draft.length <= 1}
                        onClick={deleteSelected}
                      >
                        Delete label
                      </Button>
                    </div>
                  </>
                ) : (
                  <Text font="main-ui-body" color="text-03">
                    Select a label to edit it.
                  </Text>
                )}
                {validationError && (
                  <MessageCard variant="error" title={validationError} />
                )}
                {saveError && <MessageCard variant="error" title={saveError} />}
              </div>
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          {saveError && (
            <Button
              type="button"
              prominence="tertiary"
              disabled={busy}
              onClick={() => void reload()}
            >
              {reloading ? "Reloading…" : "Reload labels"}
            </Button>
          )}
          <Button
            type="button"
            prominence="secondary"
            disabled={busy}
            onClick={onClose}
          >
            Cancel
          </Button>
          <Button
            type="button"
            disabled={!data || busy}
            onClick={() => void save()}
          >
            {saving ? "Saving…" : "Save changes"}
          </Button>
        </Modal.Footer>
      </Modal.Content>
    </Modal>
  );
}

interface FieldProps {
  label: string;
  children: React.ReactNode;
}

function Field({ label, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-1">
      <Text font="secondary-action" color="text-03">
        {label}
      </Text>
      {children}
    </div>
  );
}
