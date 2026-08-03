"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { Button, Tabs, Tag, Text } from "@opal/components";
import SvgBarChart from "@opal/icons/bar-chart";
import { SettingsLayouts, toast } from "@opal/layouts";

import MinimalMarkdown from "@/components/chat/MinimalMarkdown";
import { useProjects } from "@/lib/projects/hooks";
import {
  BenchmarkAvailableModel,
  BenchmarkCitationOption,
  BenchmarkExpectedCitationInput,
  BenchmarkQuestion,
  BenchmarkQuestionInput,
  BenchmarkRun,
  BenchmarkRunItem,
  cancelBenchmarkRun,
  createBenchmarkQuestion,
  createBenchmarkRun,
  deleteBenchmarkQuestion,
  listBenchmarkCitationOptions,
  listBenchmarkModels,
  listBenchmarkQuestions,
  listBenchmarkRuns,
  startBenchmarkRun,
  updateBenchmarkQuestion,
} from "@/lib/regulatory/benchmark";

const inputClass =
  "w-full rounded-lg border border-border-02 bg-background px-3 py-2 text-sm text-text-04 outline-none transition focus:border-action-link-05 focus:ring-2 focus:ring-action-link-01";
const panelClass = "rounded-xl border border-border-02 bg-background p-5";

type QuestionDraft = Omit<BenchmarkQuestionInput, "project_id">;

const emptyQuestion: QuestionDraft = {
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

const modelKey = (model: { provider: string; model_id: string }) =>
  `${model.provider}::${model.model_id}`;

const formatPercent = (value: number | null) =>
  value == null ? "—" : `${Math.round(value * 100)}%`;

const formatCost = (value: number | null) =>
  value == null ? "Unavailable" : `$${(value / 100).toFixed(4)}`;

const formatDuration = (value: number | null) =>
  value == null ? "—" : `${(value / 1000).toFixed(1)}s`;

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium text-text-04">{label}</span>
      {hint && <span className="text-xs text-text-02">{hint}</span>}
      {children}
    </label>
  );
}

function StatusBadge({ status }: { status: string }) {
  const color =
    status === "completed"
      ? "bg-status-success-01 text-status-success-05"
      : status === "error"
        ? "bg-status-error-01 text-status-error-05"
        : status === "running"
          ? "bg-status-info-01 text-status-info-05"
          : "bg-background-02 text-text-03";
  return (
    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${color}`}>
      {status}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-28 rounded-lg border border-border-01 bg-background-01 px-3 py-2">
      <div className="text-xs text-text-02">{label}</div>
      <div className="mt-1 text-base font-semibold text-text-05">{value}</div>
    </div>
  );
}

function QuestionsPanel() {
  const { projects } = useProjects();
  const [questions, setQuestions] = useState<BenchmarkQuestion[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<QuestionDraft>(emptyQuestion);
  const [projectId, setProjectId] = useState<string>("");
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

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!projectId) {
      setCitationOptions([]);
      return;
    }
    void listBenchmarkCitationOptions(Number(projectId))
      .then(setCitationOptions)
      .catch((error: unknown) =>
        toast.error(
          error instanceof Error ? error.message : "Citations could not load."
        )
      );
  }, [projectId]);

  const optionMap = useMemo(
    () => new Map(citationOptions.map((option) => [option.chunk_id, option])),
    [citationOptions]
  );

  const availableCitations = useMemo(() => {
    const selected = new Set(
      form.expected_citations.map((citation) => citation.chunk_id)
    );
    const query = citationSearch.trim().toLocaleLowerCase("tr");
    return citationOptions
      .filter((option) => !selected.has(option.chunk_id))
      .filter((option) => {
        if (!query) return true;
        return `${option.file_name} ${option.heading_path.join(" ")} ${option.text_excerpt}`
          .toLocaleLowerCase("tr")
          .includes(query);
      })
      .slice(0, 30);
  }, [citationOptions, citationSearch, form.expected_citations]);

  const reset = useCallback(() => {
    setEditingId(null);
    setForm(emptyQuestion);
    setFactsText("");
    setTagsText("");
    setProjectId("");
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
    setProjectId(String(question.project_id));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const updateCitation = (
    chunkId: string,
    patch: Partial<BenchmarkExpectedCitationInput>
  ) => {
    setForm((current) => ({
      ...current,
      expected_citations: current.expected_citations.map((citation) =>
        citation.chunk_id === chunkId ? { ...citation, ...patch } : citation
      ),
    }));
  };

  const save = useCallback(async () => {
    if (!projectId || !form.title.trim() || !form.prompt.trim()) return;
    setSaving(true);
    const input: BenchmarkQuestionInput = {
      ...form,
      title: form.title.trim(),
      prompt: form.prompt.trim(),
      reference_answer: form.reference_answer?.trim() || null,
      rubric_notes: form.rubric_notes?.trim() || null,
      project_id: Number(projectId),
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
  }, [editingId, factsText, form, projectId, refresh, reset, tagsText]);

  return (
    <div className="flex flex-col gap-6">
      <section className={panelClass}>
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-text-05">
              {editingId === null
                ? "Create benchmark question"
                : `Edit question #${editingId}`}
            </h2>
            <p className="mt-1 text-sm text-text-02">
              Define the gold answer, required facts, exact regulatory
              citations, temporal scope, and judge guidance.
            </p>
          </div>
          {editingId !== null && (
            <Button prominence="secondary" onClick={reset}>
              Cancel edit
            </Button>
          )}
        </div>

        <div className="grid gap-5 xl:grid-cols-2">
          <div className="flex flex-col gap-4">
            <Field label="Question title">
              <input
                className={inputClass}
                value={form.title}
                onChange={(event) =>
                  setForm({ ...form, title: event.target.value })
                }
                placeholder="e.g. Import duty exemption — Article 2"
              />
            </Field>
            <Field
              label="Directory"
              hint="The production chat run is scoped here."
            >
              <select
                className={inputClass}
                value={projectId}
                onChange={(event) => {
                  setProjectId(event.target.value);
                  setForm((current) => ({
                    ...current,
                    expected_citations: [],
                  }));
                }}
              >
                <option value="">Select directory</option>
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Question prompt">
              <textarea
                className={inputClass}
                rows={6}
                value={form.prompt}
                onChange={(event) =>
                  setForm({ ...form, prompt: event.target.value })
                }
                placeholder="Write the exact question that will be sent through normal chat."
              />
            </Field>
            <Field
              label="Gold / expected answer"
              hint="The judge compares model answers against this answer."
            >
              <textarea
                className={inputClass}
                rows={9}
                value={form.reference_answer ?? ""}
                onChange={(event) =>
                  setForm({
                    ...form,
                    reference_answer: event.target.value || null,
                  })
                }
                placeholder="Write the complete expected answer, including qualifications and dates."
              />
            </Field>
          </div>

          <div className="flex flex-col gap-4">
            <Field
              label="Expected facts"
              hint="One independently judgeable requirement per line."
            >
              <textarea
                className={inputClass}
                rows={6}
                value={factsText}
                onChange={(event) => setFactsText(event.target.value)}
                placeholder={
                  "The exemption applies only to …\nThe effective date is …"
                }
              />
            </Field>
            <div className="rounded-lg border border-border-01 bg-background-01 p-4">
              <div className="mb-3">
                <div className="text-sm font-medium text-text-04">
                  Expected citations
                </div>
                <div className="text-xs text-text-02">
                  Select exact chunks from the directory. Required citations
                  affect deterministic recall and judge scoring.
                </div>
              </div>
              {!projectId ? (
                <div className="text-sm text-text-02">
                  Select a directory to browse its chunks.
                </div>
              ) : (
                <>
                  <input
                    className={inputClass}
                    value={citationSearch}
                    onChange={(event) => setCitationSearch(event.target.value)}
                    placeholder="Search file, article, heading, or chunk text"
                  />
                  {citationSearch && availableCitations.length > 0 && (
                    <div className="mt-2 max-h-56 overflow-y-auto rounded-lg border border-border-02 bg-background">
                      {availableCitations.map((option) => (
                        <button
                          type="button"
                          key={option.chunk_id}
                          className="flex w-full flex-col gap-1 border-b border-border-01 px-3 py-2 text-left last:border-0 hover:bg-background-01"
                          onClick={() => {
                            setForm((current) => ({
                              ...current,
                              expected_citations: [
                                ...current.expected_citations,
                                {
                                  chunk_id: option.chunk_id,
                                  requirement: "required",
                                  notes: null,
                                },
                              ],
                            }));
                            setCitationSearch("");
                          }}
                        >
                          <span className="text-sm font-medium text-text-04">
                            {option.file_name} ·{" "}
                            {option.heading_path.join(" › ")}
                          </span>
                          <span className="line-clamp-2 text-xs text-text-02">
                            {option.text_excerpt}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </>
              )}
              <div className="mt-3 flex flex-col gap-2">
                {form.expected_citations.map((citation) => {
                  const option = optionMap.get(citation.chunk_id);
                  return (
                    <div
                      key={citation.chunk_id}
                      className="rounded-lg border border-border-02 bg-background p-3"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-sm font-medium text-text-04">
                            {option?.file_name ?? citation.chunk_id}
                          </div>
                          <div className="truncate text-xs text-text-02">
                            {option?.heading_path.join(" › ") ??
                              "Saved citation"}
                          </div>
                        </div>
                        <button
                          type="button"
                          className="text-xs font-medium text-status-error-05"
                          onClick={() =>
                            setForm((current) => ({
                              ...current,
                              expected_citations:
                                current.expected_citations.filter(
                                  (item) => item.chunk_id !== citation.chunk_id
                                ),
                            }))
                          }
                        >
                          Remove
                        </button>
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-[150px_1fr]">
                        <select
                          className={inputClass}
                          value={citation.requirement}
                          onChange={(event) =>
                            updateCitation(citation.chunk_id, {
                              requirement: event.target.value as
                                | "required"
                                | "supporting",
                            })
                          }
                        >
                          <option value="required">Required</option>
                          <option value="supporting">Supporting</option>
                        </select>
                        <input
                          className={inputClass}
                          value={citation.notes ?? ""}
                          onChange={(event) =>
                            updateCitation(citation.chunk_id, {
                              notes: event.target.value || null,
                            })
                          }
                          placeholder="Why this citation is expected (optional)"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <Field
                label="As-of date"
                hint="Pins temporal retrieval for repeatability."
              >
                <input
                  type="date"
                  className={inputClass}
                  value={form.as_of_date ?? ""}
                  onChange={(event) =>
                    setForm({
                      ...form,
                      as_of_date: event.target.value || null,
                    })
                  }
                />
              </Field>
              <Field label="Tags" hint="Comma separated">
                <input
                  className={inputClass}
                  value={tagsText}
                  onChange={(event) => setTagsText(event.target.value)}
                  placeholder="customs, exemption, hard"
                />
              </Field>
            </div>
            <Field
              label="Judge instructions"
              hint="Question-specific rubric, edge cases, or acceptable variants."
            >
              <textarea
                className={inputClass}
                rows={4}
                value={form.rubric_notes ?? ""}
                onChange={(event) =>
                  setForm({
                    ...form,
                    rubric_notes: event.target.value || null,
                  })
                }
                placeholder="Do not award full correctness unless …"
              />
            </Field>
          </div>
        </div>
        <div className="mt-5 flex justify-end">
          <Button
            onClick={() => void save()}
            disabled={
              saving || !projectId || !form.title.trim() || !form.prompt.trim()
            }
          >
            {saving
              ? "Saving…"
              : editingId === null
                ? "Create question"
                : "Save changes"}
          </Button>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-text-05">
            Question library
          </h2>
          <span className="text-sm text-text-02">
            {questions.length} questions
          </span>
        </div>
        {loading && <div className={panelClass}>Loading questions…</div>}
        {!loading && questions.length === 0 && (
          <div className={panelClass}>No benchmark questions yet.</div>
        )}
        {questions.map((question) => (
          <details key={question.id} className={panelClass}>
            <summary className="cursor-pointer list-none">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="font-semibold text-text-05">
                    {question.title}
                  </div>
                  <div className="mt-1 line-clamp-2 text-sm text-text-03">
                    {question.prompt}
                  </div>
                  <div className="mt-2 text-xs text-text-02">
                    {question.project_name}
                    {question.as_of_date
                      ? ` · as of ${question.as_of_date}`
                      : ""}
                    {` · ${question.expected_facts.length} facts · ${question.expected_citations.length} citations`}
                  </div>
                </div>
                <div className="flex flex-wrap justify-end gap-1.5">
                  <Tag title={question.is_active ? "active" : "inactive"} />
                  {question.tags.slice(0, 4).map((tag) => (
                    <Tag key={tag} title={tag} />
                  ))}
                </div>
              </div>
            </summary>
            <div className="mt-4 border-t border-border-01 pt-4">
              {question.reference_answer && (
                <div className="mb-4">
                  <div className="mb-1 text-xs font-semibold uppercase text-text-02">
                    Gold answer
                  </div>
                  <MinimalMarkdown content={question.reference_answer} />
                </div>
              )}
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <div className="mb-1 text-xs font-semibold uppercase text-text-02">
                    Expected facts
                  </div>
                  <ul className="list-disc pl-5 text-sm text-text-03">
                    {question.expected_facts.map((fact) => (
                      <li key={fact}>{fact}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <div className="mb-1 text-xs font-semibold uppercase text-text-02">
                    Expected citations
                  </div>
                  <ul className="list-disc pl-5 text-sm text-text-03">
                    {question.expected_citations.map((citation) => (
                      <li key={citation.chunk_id}>
                        {citation.file_name} ·{" "}
                        {citation.heading_path.join(" › ")} (
                        {citation.requirement})
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <Button
                  size="sm"
                  prominence="secondary"
                  onClick={() => edit(question)}
                >
                  Edit
                </Button>
                <Button
                  size="sm"
                  prominence="secondary"
                  onClick={() =>
                    void updateBenchmarkQuestion(question.id, {
                      is_active: !question.is_active,
                    }).then(refresh)
                  }
                >
                  {question.is_active ? "Deactivate" : "Activate"}
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  prominence="secondary"
                  onClick={() =>
                    void deleteBenchmarkQuestion(question.id)
                      .then(refresh)
                      .catch((error: unknown) =>
                        toast.error(
                          error instanceof Error
                            ? error.message
                            : "Delete failed."
                        )
                      )
                  }
                >
                  Delete
                </Button>
              </div>
            </div>
          </details>
        ))}
      </section>
    </div>
  );
}

function JudgeReport({ item }: { item: BenchmarkRunItem }) {
  const report = item.judgment?.report;
  if (!item.judgment || !report) return null;
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        <Metric label="Overall" value={`${item.judgment.overall_score}/100`} />
        <Metric
          label="Correctness"
          value={`${item.judgment.correctness_score}/5`}
        />
        <Metric
          label="Groundedness"
          value={`${item.judgment.groundedness_score}/5`}
        />
        <Metric
          label="Completeness"
          value={`${item.judgment.completeness_score}/5`}
        />
        <Metric label="Clarity" value={`${item.judgment.clarity_score}/5`} />
      </div>
      <div className="rounded-lg border border-border-01 bg-background-01 p-4">
        <div className="font-medium text-text-05">
          {report.summary ?? item.judgment.rationale}
        </div>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div>
            <div className="text-xs font-semibold uppercase text-status-success-05">
              Strengths
            </div>
            <ul className="mt-1 list-disc pl-5 text-sm text-text-03">
              {(report.strengths ?? []).map((value) => (
                <li key={value}>{value}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="text-xs font-semibold uppercase text-status-error-05">
              Weaknesses
            </div>
            <ul className="mt-1 list-disc pl-5 text-sm text-text-03">
              {(report.weaknesses ?? []).map((value) => (
                <li key={value}>{value}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
      {report.criteria && (
        <div className="grid gap-3 md:grid-cols-2">
          {Object.entries(report.criteria).map(([name, criterion]) => (
            <div key={name} className="rounded-lg border border-border-01 p-3">
              <div className="font-medium capitalize text-text-04">
                {name} · {criterion.score}/5
              </div>
              <p className="mt-1 text-sm text-text-03">{criterion.rationale}</p>
            </div>
          ))}
        </div>
      )}
      {(report.fact_assessments?.length ?? 0) > 0 && (
        <div>
          <div className="mb-2 text-sm font-semibold text-text-04">
            Expected fact assessment
          </div>
          <div className="overflow-x-auto rounded-lg border border-border-01">
            <table className="w-full text-left text-sm">
              <thead className="bg-background-01 text-text-02">
                <tr>
                  <th className="p-2">Fact</th>
                  <th className="p-2">Verdict</th>
                  <th className="p-2">Explanation</th>
                </tr>
              </thead>
              <tbody>
                {report.fact_assessments?.map((fact) => (
                  <tr key={fact.fact} className="border-t border-border-01">
                    <td className="p-2 text-text-04">{fact.fact}</td>
                    <td className="p-2 font-medium text-text-04">
                      {fact.verdict}
                    </td>
                    <td className="p-2 text-text-03">{fact.explanation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {(report.citation_assessments?.length ?? 0) > 0 && (
        <div>
          <div className="mb-2 text-sm font-semibold text-text-04">
            Expected citation assessment
          </div>
          <div className="flex flex-col gap-2">
            {report.citation_assessments?.map((citation) => (
              <div
                key={citation.expected_chunk_id}
                className="rounded-lg border border-border-01 p-3 text-sm"
              >
                <span className="font-medium text-text-04">
                  {citation.verdict}
                </span>
                <span className="ml-2 text-text-02">
                  {citation.expected_chunk_id}
                </span>
                <p className="mt-1 text-text-03">{citation.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ItemDetails({ item }: { item: BenchmarkRunItem }) {
  return (
    <details className={panelClass}>
      <summary className="cursor-pointer list-none">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="font-semibold text-text-05">
              {item.question_title}
            </div>
            <div className="mt-1 text-sm text-text-02">
              {item.provider} / {item.model_id}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={item.status} />
            <span className="text-sm text-text-03">
              Score {item.judgment?.overall_score ?? "—"}
            </span>
            <span className="text-sm text-text-03">
              {item.total_tokens ?? "—"} tokens
            </span>
            <span className="text-sm text-text-03">
              {formatDuration(item.duration_ms)}
            </span>
          </div>
        </div>
      </summary>
      <div className="mt-5 flex flex-col gap-6 border-t border-border-01 pt-5">
        {(item.error_message || item.judge_error) && (
          <div className="rounded-lg border border-status-error-03 bg-status-error-01 p-3 text-sm text-status-error-05">
            {item.error_message || item.judge_error}
          </div>
        )}
        <section>
          <h4 className="mb-2 text-sm font-semibold text-text-04">
            Model answer
          </h4>
          <div className="rounded-lg border border-border-01 bg-background-01 p-4">
            {item.final_result ? (
              <MinimalMarkdown content={item.final_result} />
            ) : (
              <span className="text-sm text-text-02">No answer captured.</span>
            )}
          </div>
        </section>
        <section>
          <h4 className="mb-2 text-sm font-semibold text-text-04">
            Citation performance
          </h4>
          <div className="mb-3 flex gap-2">
            <Metric
              label="Required recall"
              value={formatPercent(item.citation_recall)}
            />
            <Metric
              label="Precision"
              value={formatPercent(item.citation_precision)}
            />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {item.cited_sources.map((source, index) => (
              <div
                key={`${String(source.regulatory_chunk_id)}-${index}`}
                className="rounded-lg border border-border-01 p-3"
              >
                <div className="text-sm font-medium text-text-04">
                  [{String(source.citation_number)}]{" "}
                  {String(
                    source.file_name ?? source.semantic_identifier ?? "Source"
                  )}
                </div>
                <div className="mt-1 text-xs text-text-02">
                  {Array.isArray(source.heading_path)
                    ? source.heading_path.join(" › ")
                    : ""}
                </div>
                <p className="mt-2 line-clamp-5 text-sm text-text-03">
                  {String(source.excerpt ?? "")}
                </p>
                <div className="mt-2 break-all text-xs text-text-02">
                  {String(
                    source.regulatory_chunk_id ?? "No regulatory chunk id"
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
        <section>
          <h4 className="mb-2 text-sm font-semibold text-text-04">
            Production chat execution steps
          </h4>
          <div className="flex flex-col gap-2 border-l-2 border-border-02 pl-4">
            {item.execution_steps.map((step, index) => (
              <details
                key={index}
                className="rounded-lg border border-border-01 p-3"
              >
                <summary className="cursor-pointer text-sm font-medium text-text-04">
                  {index + 1}. {String(step.title ?? step.kind ?? "Step")}
                </summary>
                <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-background-01 p-3 text-xs text-text-03">
                  {JSON.stringify(step, null, 2)}
                </pre>
              </details>
            ))}
          </div>
        </section>
        <section>
          <h4 className="mb-2 text-sm font-semibold text-text-04">
            LLM calls and usage
          </h4>
          <div className="overflow-x-auto rounded-lg border border-border-01">
            <table className="w-full text-left text-sm">
              <thead className="bg-background-01 text-text-02">
                <tr>
                  <th className="p-2">Phase</th>
                  <th className="p-2">Model</th>
                  <th className="p-2">Input</th>
                  <th className="p-2">Output</th>
                  <th className="p-2">Cache</th>
                </tr>
              </thead>
              <tbody>
                {item.llm_calls.map((call, index) => (
                  <tr key={index} className="border-t border-border-01">
                    <td className="p-2">{String(call.phase ?? "answer")}</td>
                    <td className="p-2">{String(call.model ?? "—")}</td>
                    <td className="p-2">{String(call.input_tokens ?? 0)}</td>
                    <td className="p-2">{String(call.output_tokens ?? 0)}</td>
                    <td className="p-2">
                      {String(call.cache_read_tokens ?? 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-2 text-xs text-text-02">
            Answer cost: {formatCost(item.cost_cents)} · Judge cost:{" "}
            {formatCost(item.judgment?.cost_cents ?? null)} · Chat session:{" "}
            {item.chat_session_id ?? "—"}
          </div>
        </section>
        {item.answer_reasoning && (
          <details>
            <summary className="cursor-pointer text-sm font-semibold text-text-04">
              Model-provided reasoning
            </summary>
            <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-border-01 bg-background-01 p-4 text-sm text-text-03">
              {item.answer_reasoning}
            </pre>
          </details>
        )}
        <section>
          <h4 className="mb-2 text-sm font-semibold text-text-04">
            Judge report
          </h4>
          <JudgeReport item={item} />
        </section>
      </div>
    </details>
  );
}

function RunsPanel() {
  const [questions, setQuestions] = useState<BenchmarkQuestion[]>([]);
  const [models, setModels] = useState<BenchmarkAvailableModel[]>([]);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [selectedQuestionIds, setSelectedQuestionIds] = useState<number[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [judgeKey, setJudgeKey] = useState<string>("");
  const [modelSearch, setModelSearch] = useState("");
  const [judgeSearch, setJudgeSearch] = useState("");
  const [questionSearch, setQuestionSearch] = useState("");
  const [label, setLabel] = useState("");
  const [deepResearch, setDeepResearch] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [launching, setLaunching] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [nextQuestions, nextRuns, nextModels] = await Promise.all([
        listBenchmarkQuestions(),
        listBenchmarkRuns(),
        listBenchmarkModels(),
      ]);
      setQuestions(nextQuestions.filter((question) => question.is_active));
      setRuns(nextRuns);
      setModels(nextModels);
      setSelectedRunId((current) => current ?? nextRuns[0]?.id ?? null);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Benchmark data failed."
      );
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    if (
      !runs.some((run) => run.status === "pending" || run.status === "running")
    )
      return;
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
  }, [refresh, runs]);

  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const modelMap = useMemo(
    () => new Map(models.map((model) => [modelKey(model), model])),
    [models]
  );
  const filteredModels = useMemo(() => {
    const query = modelSearch.trim().toLowerCase();
    return models
      .filter(
        (model) =>
          !query ||
          `${model.display_name} ${model.model_id}`
            .toLowerCase()
            .includes(query)
      )
      .slice(0, 80);
  }, [modelSearch, models]);
  const filteredJudgeModels = useMemo(() => {
    const query = judgeSearch.trim().toLowerCase();
    return models.filter(
      (model) =>
        !query ||
        `${model.display_name} ${model.model_id}`.toLowerCase().includes(query)
    );
  }, [judgeSearch, models]);
  const filteredQuestions = useMemo(() => {
    const query = questionSearch.trim().toLocaleLowerCase("tr");
    return questions.filter(
      (question) =>
        !query ||
        `${question.title} ${question.prompt} ${question.tags.join(" ")}`
          .toLocaleLowerCase("tr")
          .includes(query)
    );
  }, [questionSearch, questions]);

  const launch = useCallback(async () => {
    const judge = modelMap.get(judgeKey);
    const candidates = selectedModels
      .map((key) => modelMap.get(key))
      .filter((model): model is BenchmarkAvailableModel => Boolean(model));
    if (!judge || candidates.length === 0 || selectedQuestionIds.length === 0)
      return;
    setLaunching(true);
    try {
      const run = await createBenchmarkRun({
        label: label.trim() || null,
        question_ids: selectedQuestionIds,
        candidates: candidates.map(({ provider, model_id }) => ({
          provider,
          model_id,
        })),
        judge: { provider: judge.provider, model_id: judge.model_id },
        deep_research: deepResearch,
      });
      await startBenchmarkRun(run.id);
      setSelectedRunId(run.id);
      toast.success("Benchmark run queued through the production chat flow.");
      await refresh();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Run launch failed."
      );
    } finally {
      setLaunching(false);
    }
  }, [
    deepResearch,
    judgeKey,
    label,
    modelMap,
    refresh,
    selectedModels,
    selectedQuestionIds,
  ]);

  return (
    <div className="flex flex-col gap-6">
      <section className={panelClass}>
        <div className="mb-5">
          <h2 className="text-lg font-semibold text-text-05">
            Create benchmark run
          </h2>
          <p className="mt-1 text-sm text-text-02">
            Every item calls the exact production chat generator. Choose
            explicit questions, OpenRouter candidate models, and an independent
            OpenRouter judge.
          </p>
        </div>
        <div className="grid items-start gap-6 xl:grid-cols-3">
          <div className="flex min-w-0 flex-col gap-3">
            <Field label="Run label">
              <input
                className={inputClass}
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                placeholder="July legal QA baseline"
              />
            </Field>
            <Field label={`Questions · ${selectedQuestionIds.length} selected`}>
              <input
                className={inputClass}
                value={questionSearch}
                onChange={(event) => setQuestionSearch(event.target.value)}
                placeholder="Search questions"
              />
            </Field>
            <div className="flex gap-2">
              <button
                type="button"
                className="text-xs font-medium text-action-link-05"
                onClick={() =>
                  setSelectedQuestionIds(
                    filteredQuestions.map((question) => question.id)
                  )
                }
              >
                Select visible
              </button>
              <button
                type="button"
                className="text-xs font-medium text-text-02"
                onClick={() => setSelectedQuestionIds([])}
              >
                Clear
              </button>
            </div>
            <div className="h-72 overflow-y-auto rounded-lg border border-border-02">
              {filteredQuestions.length === 0 ? (
                <div className="flex h-full items-center justify-center p-4 text-center text-sm text-text-02">
                  No matching questions.
                </div>
              ) : (
                filteredQuestions.map((question) => (
                  <label
                    key={question.id}
                    className="flex cursor-pointer gap-2 border-b border-border-01 p-3 last:border-0 hover:bg-background-01"
                  >
                    <input
                      type="checkbox"
                      checked={selectedQuestionIds.includes(question.id)}
                      onChange={(event) =>
                        setSelectedQuestionIds((current) =>
                          event.target.checked
                            ? current.includes(question.id)
                              ? current
                              : [...current, question.id]
                            : current.filter((id) => id !== question.id)
                        )
                      }
                    />
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-medium text-text-04">
                        {question.title}
                      </span>
                      <span className="line-clamp-1 text-xs text-text-02">
                        {question.prompt}
                      </span>
                    </span>
                  </label>
                ))
              )}
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-3">
            <Field
              label={`Candidate models · ${selectedModels.length} selected`}
              hint={`${models.length} OpenRouter models available`}
            >
              <input
                className={inputClass}
                value={modelSearch}
                onChange={(event) => setModelSearch(event.target.value)}
                placeholder="Search OpenRouter catalog"
              />
            </Field>
            <div className="flex h-72 min-h-0 flex-col overflow-hidden rounded-lg border border-border-02">
              {selectedModels.length > 0 && (
                <div className="max-h-24 shrink-0 overflow-y-auto border-b border-border-02 bg-background-01 p-2">
                  <div className="flex flex-wrap gap-1.5">
                    {selectedModels.map((key) => {
                      const model = modelMap.get(key);
                      const displayName = model?.display_name ?? key;
                      return (
                        <button
                          type="button"
                          key={key}
                          title={`Remove ${displayName}`}
                          aria-label={`Remove ${displayName}`}
                          className="flex max-w-56 items-center gap-1 rounded-full border border-border-02 bg-background px-2 py-1 text-xs text-text-04 hover:bg-background-02"
                          onClick={() =>
                            setSelectedModels((current) =>
                              current.filter((item) => item !== key)
                            )
                          }
                        >
                          <span className="truncate">{displayName}</span>
                          <span aria-hidden="true" className="shrink-0">
                            ×
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              <div className="min-h-0 flex-1 overflow-y-auto">
                {filteredModels.length === 0 ? (
                  <div className="flex h-full items-center justify-center p-4 text-center text-sm text-text-02">
                    No matching OpenRouter models.
                  </div>
                ) : (
                  filteredModels.map((model) => {
                    const key = modelKey(model);
                    return (
                      <label
                        key={key}
                        className="flex cursor-pointer items-start gap-2 border-b border-border-01 p-3 last:border-0 hover:bg-background-01"
                      >
                        <input
                          type="checkbox"
                          checked={selectedModels.includes(key)}
                          onChange={(event) =>
                            setSelectedModels((current) =>
                              event.target.checked
                                ? current.includes(key)
                                  ? current
                                  : [...current, key]
                                : current.filter((item) => item !== key)
                            )
                          }
                        />
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-medium text-text-04">
                            {model.display_name}
                          </span>
                          <span className="block truncate text-xs text-text-02">
                            {model.model_id}
                            {model.max_input_tokens
                              ? ` · ${model.max_input_tokens.toLocaleString()} context`
                              : ""}
                          </span>
                        </span>
                      </label>
                    );
                  })
                )}
              </div>
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-4">
            <Field
              label="Judge model"
              hint="Use a strong model; it scores every answer and writes the final report."
            >
              <input
                className={inputClass}
                value={judgeSearch}
                onChange={(event) => setJudgeSearch(event.target.value)}
                placeholder="Filter judge models"
              />
              <select
                className={`${inputClass} mt-2`}
                value={judgeKey}
                onChange={(event) => setJudgeKey(event.target.value)}
              >
                <option value="">Select OpenRouter judge</option>
                {filteredJudgeModels.map((model) => (
                  <option key={modelKey(model)} value={modelKey(model)}>
                    {model.display_name} — {model.model_id}
                  </option>
                ))}
              </select>
            </Field>
            <label className="flex cursor-pointer items-start gap-3 rounded-lg border border-border-02 p-3">
              <input
                type="checkbox"
                className="mt-1"
                checked={deepResearch}
                onChange={(event) => setDeepResearch(event.target.checked)}
              />
              <span>
                <span className="block text-sm font-medium text-text-04">
                  Deep research
                </span>
                <span className="block text-xs text-text-02">
                  Uses the production deep-research loop; no web-search tool is
                  enabled.
                </span>
              </span>
            </label>
            <div className="rounded-lg bg-background-01 p-3 text-sm text-text-03">
              <div>
                {selectedQuestionIds.length} questions × {selectedModels.length}{" "}
                models
              </div>
              <div className="mt-1 font-semibold text-text-05">
                {selectedQuestionIds.length * selectedModels.length} total run
                items
              </div>
            </div>
            <Button
              disabled={
                launching ||
                selectedQuestionIds.length === 0 ||
                selectedModels.length === 0 ||
                !judgeKey
              }
              onClick={() => void launch()}
            >
              {launching ? "Launching…" : "Create and start run"}
            </Button>
          </div>
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-[300px_1fr]">
        <aside className="flex flex-col gap-2">
          <div className="mb-1 text-sm font-semibold text-text-04">
            Run history
          </div>
          {runs.map((run) => (
            <button
              type="button"
              key={run.id}
              className={`rounded-lg border p-3 text-left ${run.id === selectedRunId ? "border-action-link-05 bg-action-link-01" : "border-border-02 bg-background"}`}
              onClick={() => setSelectedRunId(run.id)}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm font-medium text-text-04">
                  {run.label ?? `Run #${run.id}`}
                </span>
                <StatusBadge status={run.status} />
              </div>
              <div className="mt-2 text-xs text-text-02">
                {run.completed_items + run.failed_items}/{run.total_items} items
                · {run.judge_model}
              </div>
            </button>
          ))}
        </aside>
        <main className="min-w-0">
          {!selectedRun && (
            <div className={panelClass}>
              Select a run to inspect its results.
            </div>
          )}
          {selectedRun && (
            <div className="flex flex-col gap-4">
              <div className={panelClass}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h2 className="text-xl font-semibold text-text-05">
                      {selectedRun.label ?? `Run #${selectedRun.id}`}
                    </h2>
                    <p className="mt-1 text-sm text-text-02">
                      Judge: {selectedRun.judge_provider} /{" "}
                      {selectedRun.judge_model} ·{" "}
                      {selectedRun.deep_research
                        ? "Deep research"
                        : "Internal search"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <StatusBadge status={selectedRun.status} />
                    {(selectedRun.status === "pending" ||
                      selectedRun.status === "running") && (
                      <Button
                        size="sm"
                        variant="danger"
                        prominence="secondary"
                        onClick={() =>
                          void cancelBenchmarkRun(selectedRun.id).then(refresh)
                        }
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                </div>
                <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                  <Metric
                    label="Progress"
                    value={`${selectedRun.completed_items + selectedRun.failed_items}/${selectedRun.total_items}`}
                  />
                  <Metric
                    label="Completed"
                    value={String(selectedRun.completed_items)}
                  />
                  <Metric
                    label="Failed"
                    value={String(selectedRun.failed_items)}
                  />
                  <Metric
                    label="Report cost"
                    value={formatCost(selectedRun.report_cost_cents)}
                  />
                </div>
              </div>
              {selectedRun.report && (
                <details open className={panelClass}>
                  <summary className="cursor-pointer text-lg font-semibold text-text-05">
                    Judge comparison report
                  </summary>
                  <div className="mt-4">
                    <p className="text-sm text-text-03">
                      {selectedRun.report.executive_summary}
                    </p>
                    <div className="mt-4 grid gap-3 md:grid-cols-2">
                      {selectedRun.report.model_reports
                        .sort((a, b) => a.rank - b.rank)
                        .map((model) => (
                          <div
                            key={`${model.provider}-${model.model_id}`}
                            className="rounded-lg border border-border-01 p-3"
                          >
                            <div className="font-semibold text-text-05">
                              #{model.rank} · {model.model_id}
                            </div>
                            <p className="mt-1 text-sm text-text-03">
                              {model.summary}
                            </p>
                            <div className="mt-2 text-xs font-medium text-text-04">
                              Recommended use
                            </div>
                            <p className="text-sm text-text-03">
                              {model.recommended_use}
                            </p>
                          </div>
                        ))}
                    </div>
                    <div className="mt-4 rounded-lg bg-background-01 p-3">
                      <div className="text-sm font-semibold text-text-04">
                        Recommendation
                      </div>
                      <p className="mt-1 text-sm text-text-03">
                        {selectedRun.report.recommendation}
                      </p>
                    </div>
                  </div>
                </details>
              )}
              {selectedRun.report_error && (
                <div className="rounded-lg border border-status-error-03 bg-status-error-01 p-3 text-sm text-status-error-05">
                  Run report failed: {selectedRun.report_error}
                </div>
              )}
              <section className="grid gap-3 md:grid-cols-2">
                {selectedRun.aggregates.map((aggregate) => (
                  <div
                    key={`${aggregate.provider}:${aggregate.model_id}`}
                    className={panelClass}
                  >
                    <div className="font-semibold text-text-05">
                      {aggregate.model_id}
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                      <div className="text-text-02">Score</div>
                      <div className="text-right font-medium text-text-04">
                        {aggregate.average_score?.toFixed(1) ?? "—"}
                      </div>
                      <div className="text-text-02">Citation recall</div>
                      <div className="text-right font-medium text-text-04">
                        {formatPercent(aggregate.average_citation_recall)}
                      </div>
                      <div className="text-text-02">Avg tokens</div>
                      <div className="text-right font-medium text-text-04">
                        {aggregate.average_tokens?.toFixed(0) ?? "—"}
                      </div>
                      <div className="text-text-02">Avg latency</div>
                      <div className="text-right font-medium text-text-04">
                        {formatDuration(aggregate.average_duration_ms)}
                      </div>
                      <div className="text-text-02">Answer + judge cost</div>
                      <div className="text-right font-medium text-text-04">
                        {formatCost(aggregate.total_cost_cents)}
                      </div>
                    </div>
                  </div>
                ))}
              </section>
              <section className="flex flex-col gap-3">
                <h3 className="text-lg font-semibold text-text-05">
                  Item results
                </h3>
                {selectedRun.items.map((item) => (
                  <ItemDetails key={item.id} item={item} />
                ))}
              </section>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default function BenchmarkPage() {
  return (
    <SettingsLayouts.Root width="full">
      <SettingsLayouts.Header
        icon={SvgBarChart}
        title="Benchmark"
        description="Build reproducible regulatory QA evaluations, run the exact production chat flow through OpenRouter models, and inspect evidence-grounded judge reports."
        divider
      />
      <SettingsLayouts.Body>
        <Tabs defaultValue="questions" variant="pill">
          <Tabs.List>
            <Tabs.Trigger value="questions">Question library</Tabs.Trigger>
            <Tabs.Trigger value="runs">Runs & reports</Tabs.Trigger>
          </Tabs.List>
          <Tabs.Content value="questions">
            <QuestionsPanel />
          </Tabs.Content>
          <Tabs.Content value="runs">
            <RunsPanel />
          </Tabs.Content>
        </Tabs>
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
