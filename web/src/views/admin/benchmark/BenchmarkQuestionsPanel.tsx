"use client";

import { useCallback, useEffect, useState } from "react";

import { toast } from "@opal/layouts";

import { useDocumentSets } from "@/lib/hooks/useDocumentSets";
import {
  type BenchmarkCitationOption,
  type BenchmarkQuestion,
  type BenchmarkQuestionInput,
  createBenchmarkQuestion,
  listBenchmarkCitationOptions,
  listBenchmarkQuestions,
  updateBenchmarkQuestion,
} from "@/lib/regulatory/benchmark";
import BenchmarkQuestionForm, {
  type BenchmarkQuestionDraft,
} from "@/views/admin/benchmark/BenchmarkQuestionForm";
import BenchmarkQuestionLibrary from "@/views/admin/benchmark/BenchmarkQuestionLibrary";

const emptyQuestion: BenchmarkQuestionDraft = {
  title: "",
  prompt: "",
  reference_answer: null,
  expected_facts: [],
  expected_citations: [],
  as_of_date: null,
  rubric_notes: null,
  tags: [],
};

const splitLines = (value: string) =>
  value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);

export default function BenchmarkQuestionsPanel() {
  const { documentSets } = useDocumentSets();
  const [questions, setQuestions] = useState<BenchmarkQuestion[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<BenchmarkQuestionDraft>(emptyQuestion);
  const [documentSetId, setDocumentSetId] = useState("");
  const [factsText, setFactsText] = useState("");
  const [tagsText, setTagsText] = useState("");
  const [citationOptions, setCitationOptions] = useState<
    BenchmarkCitationOption[]
  >([]);
  const [citationSearch, setCitationSearch] = useState("");
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setQuestions(await listBenchmarkQuestions());
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Questions failed.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void refresh(), [refresh]);
  useEffect(() => {
    let ignoreResponse = false;
    setCitationOptions([]);
    if (!documentSetId) {
      return;
    }
    void listBenchmarkCitationOptions(Number(documentSetId))
      .then((options) => {
        if (!ignoreResponse) setCitationOptions(options);
      })
      .catch((error: unknown) => {
        if (!ignoreResponse) {
          toast.error(
            error instanceof Error ? error.message : "Citations could not load."
          );
        }
      });
    return () => {
      ignoreResponse = true;
    };
  }, [documentSetId]);

  const reset = useCallback(() => {
    setEditingId(null);
    setForm(emptyQuestion);
    setFactsText("");
    setTagsText("");
    setDocumentSetId("");
    setCitationSearch("");
  }, []);

  const edit = useCallback((question: BenchmarkQuestion) => {
    setEditingId(question.id);
    setForm({
      title: question.title,
      prompt: question.prompt,
      reference_answer: question.reference_answer,
      expected_facts: question.expected_facts,
      expected_citations: question.expected_citations.map((citation) => ({
        chunk_id: citation.chunk_id,
        requirement: citation.requirement,
        notes: citation.notes,
      })),
      as_of_date: question.as_of_date,
      rubric_notes: question.rubric_notes,
      tags: question.tags,
    });
    setFactsText(question.expected_facts.join("\n"));
    setTagsText(question.tags.join(", "));
    setDocumentSetId(String(question.document_set_id));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const save = useCallback(async () => {
    if (!documentSetId || !form.title.trim() || !form.prompt.trim()) return;
    setSaving(true);
    const input: BenchmarkQuestionInput = {
      ...form,
      title: form.title.trim(),
      prompt: form.prompt.trim(),
      reference_answer: form.reference_answer?.trim() || null,
      rubric_notes: form.rubric_notes?.trim() || null,
      document_set_id: Number(documentSetId),
      expected_facts: splitLines(factsText),
      tags: tagsText
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean),
    };
    try {
      if (editingId === null) await createBenchmarkQuestion(input);
      else await updateBenchmarkQuestion(editingId, input);
      toast.success(
        editingId === null ? "Question created." : "Question updated."
      );
      reset();
      await refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Save failed.");
    } finally {
      setSaving(false);
    }
  }, [documentSetId, editingId, factsText, form, refresh, reset, tagsText]);

  return (
    <div className="flex flex-col gap-6">
      <BenchmarkQuestionForm
        editingId={editingId}
        form={form}
        setForm={setForm}
        documentSets={documentSets}
        documentSetId={documentSetId}
        setDocumentSetId={setDocumentSetId}
        factsText={factsText}
        setFactsText={setFactsText}
        tagsText={tagsText}
        setTagsText={setTagsText}
        citationOptions={citationOptions}
        citationSearch={citationSearch}
        setCitationSearch={setCitationSearch}
        saving={saving}
        onReset={reset}
        onSave={save}
      />
      <BenchmarkQuestionLibrary
        questions={questions}
        loading={loading}
        onEdit={edit}
        onRefresh={refresh}
      />
    </div>
  );
}
