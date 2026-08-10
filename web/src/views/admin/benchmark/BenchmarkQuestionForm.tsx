"use client";

import type { Dispatch, SetStateAction } from "react";

import {
  Button,
  Card,
  InputDatePicker,
  InputSelect,
  InputTextArea,
  InputTypeIn,
  Text,
} from "@opal/components";

import type {
  BenchmarkCitationOption,
  BenchmarkQuestionInput,
} from "@/lib/regulatory/benchmark";
import BenchmarkCitationPicker from "@/views/admin/benchmark/BenchmarkCitationPicker";
import { FormField } from "@/views/admin/benchmark/BenchmarkPresentation";

export type BenchmarkQuestionDraft = Omit<
  BenchmarkQuestionInput,
  "document_set_id"
>;

interface DocumentSetOption {
  id: number;
  name: string;
}

interface BenchmarkQuestionFormProps {
  editingId: number | null;
  form: BenchmarkQuestionDraft;
  setForm: Dispatch<SetStateAction<BenchmarkQuestionDraft>>;
  documentSets: DocumentSetOption[];
  documentSetId: string;
  setDocumentSetId: (value: string) => void;
  factsText: string;
  setFactsText: (value: string) => void;
  tagsText: string;
  setTagsText: (value: string) => void;
  citationOptions: BenchmarkCitationOption[];
  citationSearch: string;
  setCitationSearch: (value: string) => void;
  saving: boolean;
  onReset: () => void;
  onSave: () => Promise<void>;
}

const parseIsoDate = (value: string | null) => {
  if (!value) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return null;
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
};

const formatIsoDate = (value: Date | null) => {
  if (!value) return null;
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const day = String(value.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

export default function BenchmarkQuestionForm({
  editingId,
  form,
  setForm,
  documentSets,
  documentSetId,
  setDocumentSetId,
  factsText,
  setFactsText,
  tagsText,
  setTagsText,
  citationOptions,
  citationSearch,
  setCitationSearch,
  saving,
  onReset,
  onSave,
}: BenchmarkQuestionFormProps) {
  return (
    <Card border="solid" padding="lg">
      <div className="mb-5 flex items-start justify-between gap-4">
        <div>
          <Text as="h2" font="heading-h3" color="text-05">
            {editingId === null
              ? "Create benchmark question"
              : `Edit question #${editingId}`}
          </Text>
          <div className="mt-1">
            <Text as="p" font="main-ui-muted" color="text-02">
              Define the gold answer, required facts, exact regulatory
              citations, temporal scope, and judge guidance.
            </Text>
          </div>
        </div>
        {editingId !== null && (
          <Button prominence="secondary" onClick={onReset}>
            Cancel edit
          </Button>
        )}
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <div className="flex flex-col gap-4">
          <FormField label="Question title">
            <InputTypeIn
              value={form.title}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  title: event.target.value,
                }))
              }
              placeholder="e.g. Import duty exemption — Article 2"
            />
          </FormField>
          <FormField
            label="Document Set"
            hint="The production chat run is scoped to this document set."
          >
            <InputSelect
              value={documentSetId}
              onValueChange={(value) => {
                setDocumentSetId(value);
                setForm((current) => ({
                  ...current,
                  expected_citations: [],
                }));
              }}
            >
              <InputSelect.Trigger
                aria-label="Document Set"
                placeholder="Select document set"
              />
              <InputSelect.Content>
                {documentSets.map((documentSet) => (
                  <InputSelect.Item
                    key={documentSet.id}
                    value={String(documentSet.id)}
                  >
                    {documentSet.name}
                  </InputSelect.Item>
                ))}
              </InputSelect.Content>
            </InputSelect>
          </FormField>
          <FormField label="Question prompt">
            <InputTextArea
              rows={6}
              value={form.prompt}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  prompt: event.target.value,
                }))
              }
              placeholder="Write the exact question that will be sent through normal chat."
            />
          </FormField>
          <FormField
            label="Gold / expected answer"
            hint="The judge compares model answers against this answer."
          >
            <InputTextArea
              rows={9}
              value={form.reference_answer ?? ""}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  reference_answer: event.target.value || null,
                }))
              }
              placeholder="Write the complete expected answer, including qualifications and dates."
            />
          </FormField>
        </div>

        <div className="flex flex-col gap-4">
          <FormField
            label="Expected facts"
            hint="One independently judgeable requirement per line."
          >
            <InputTextArea
              rows={6}
              value={factsText}
              onChange={(event) => setFactsText(event.target.value)}
              placeholder={
                "The exemption applies only to …\nThe effective date is …"
              }
            />
          </FormField>
          <BenchmarkCitationPicker
            documentSetId={documentSetId}
            citations={form.expected_citations}
            options={citationOptions}
            search={citationSearch}
            onSearchChange={setCitationSearch}
            onChange={(expectedCitations) =>
              setForm((current) => ({
                ...current,
                expected_citations: expectedCitations,
              }))
            }
          />
          <div className="grid gap-4 md:grid-cols-2">
            <FormField
              label="As-of date"
              hint="Pins temporal retrieval for repeatability."
            >
              <InputDatePicker
                id="benchmark-as-of-date"
                value={parseIsoDate(form.as_of_date)}
                onChange={(date) =>
                  setForm((current) => ({
                    ...current,
                    as_of_date: formatIsoDate(date),
                  }))
                }
              />
            </FormField>
            <FormField label="Tags" hint="Comma separated">
              <InputTypeIn
                value={tagsText}
                onChange={(event) => setTagsText(event.target.value)}
                placeholder="customs, exemption, hard"
              />
            </FormField>
          </div>
          <FormField
            label="Judge instructions"
            hint="Question-specific rubric, edge cases, or acceptable variants."
          >
            <InputTextArea
              rows={4}
              value={form.rubric_notes ?? ""}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  rubric_notes: event.target.value || null,
                }))
              }
              placeholder="Do not award full correctness unless …"
            />
          </FormField>
        </div>
      </div>
      <div className="mt-5 flex justify-end">
        <Button
          onClick={() => void onSave()}
          disabled={
            saving ||
            !documentSetId ||
            !form.title.trim() ||
            !form.prompt.trim()
          }
        >
          {saving
            ? "Saving…"
            : editingId === null
              ? "Create question"
              : "Save changes"}
        </Button>
      </div>
    </Card>
  );
}
