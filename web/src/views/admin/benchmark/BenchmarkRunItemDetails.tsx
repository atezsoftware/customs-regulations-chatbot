"use client";

import { Text } from "@opal/components";

import MinimalMarkdown from "@/components/chat/MinimalMarkdown";
import type { BenchmarkRunItem } from "@/lib/regulatory/benchmark";
import {
  formatCost,
  formatDuration,
  formatPercent,
  Metric,
  StatusBadge,
} from "@/views/admin/benchmark/BenchmarkPresentation";

const panelClass = "rounded-xl border border-border-02 bg-background p-5";

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
        <Text font="main-ui-action" color="text-05">
          {report.summary ?? item.judgment.rationale}
        </Text>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div>
            <Text font="secondary-action" color="status-success-05">
              Strengths
            </Text>
            <ul className="mt-1 list-disc pl-5">
              {(report.strengths ?? []).map((value) => (
                <Text as="li" key={value} font="main-ui-body" color="text-03">
                  {value}
                </Text>
              ))}
            </ul>
          </div>
          <div>
            <Text font="secondary-action" color="status-error-05">
              Weaknesses
            </Text>
            <ul className="mt-1 list-disc pl-5">
              {(report.weaknesses ?? []).map((value) => (
                <Text as="li" key={value} font="main-ui-body" color="text-03">
                  {value}
                </Text>
              ))}
            </ul>
          </div>
        </div>
      </div>
      {report.criteria && (
        <div className="grid gap-3 md:grid-cols-2">
          {Object.entries(report.criteria).map(([name, criterion]) => (
            <div key={name} className="rounded-lg border border-border-01 p-3">
              <Text font="main-ui-action" color="text-04">
                {`${name} · ${criterion.score}/5`}
              </Text>
              <div className="mt-1">
                <Text as="p" font="main-ui-body" color="text-03">
                  {criterion.rationale}
                </Text>
              </div>
            </div>
          ))}
        </div>
      )}
      {(report.fact_assessments?.length ?? 0) > 0 && (
        <div>
          <div className="mb-2">
            <Text font="main-ui-action" color="text-04">
              Expected fact assessment
            </Text>
          </div>
          <div className="overflow-x-auto rounded-lg border border-border-01">
            <table className="w-full text-left text-sm">
              <thead className="bg-background-01 text-text-02">
                <tr>
                  {(["Fact", "Verdict", "Explanation"] as const).map(
                    (label) => (
                      <th key={label} className="p-2">
                        <Text font="secondary-action" color="text-02">
                          {label}
                        </Text>
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {report.fact_assessments?.map((fact) => (
                  <tr key={fact.fact} className="border-t border-border-01">
                    <td className="p-2">
                      <Text font="main-ui-body" color="text-04">
                        {fact.fact}
                      </Text>
                    </td>
                    <td className="p-2">
                      <Text font="main-ui-action" color="text-04">
                        {fact.verdict}
                      </Text>
                    </td>
                    <td className="p-2">
                      <Text font="main-ui-body" color="text-03">
                        {fact.explanation}
                      </Text>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {(report.citation_assessments?.length ?? 0) > 0 && (
        <div>
          <div className="mb-2">
            <Text font="main-ui-action" color="text-04">
              Expected citation assessment
            </Text>
          </div>
          <div className="flex flex-col gap-2">
            {report.citation_assessments?.map((citation) => (
              <div
                key={citation.expected_chunk_id}
                className="rounded-lg border border-border-01 p-3 text-sm"
              >
                <Text font="main-ui-action" color="text-04">
                  {citation.verdict}
                </Text>
                <Text font="secondary-body" color="text-02">
                  {` ${citation.expected_chunk_id}`}
                </Text>
                <div className="mt-1">
                  <Text as="p" font="main-ui-body" color="text-03">
                    {citation.explanation}
                  </Text>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function BenchmarkRunItemDetails({
  item,
}: {
  item: BenchmarkRunItem;
}) {
  return (
    <details className={panelClass}>
      <summary className="cursor-pointer list-none">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <Text font="main-ui-action" color="text-05">
              {item.question_title}
            </Text>
            <div className="mt-1">
              <Text font="main-ui-body" color="text-02">
                {`${item.provider} / ${item.model_id}`}
              </Text>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={item.status} />
            <Text font="main-ui-body" color="text-03">
              {`Score ${item.judgment?.overall_score ?? "—"}`}
            </Text>
            <Text font="main-ui-body" color="text-03">
              {`${item.total_tokens ?? "—"} tokens`}
            </Text>
            <Text font="main-ui-body" color="text-03">
              {formatDuration(item.duration_ms)}
            </Text>
          </div>
        </div>
      </summary>
      <div className="mt-5 flex flex-col gap-6 border-t border-border-01 pt-5">
        {(item.error_message || item.judge_error) && (
          <div className="rounded-lg border border-status-error-03 bg-status-error-01 p-3">
            <Text font="main-ui-body" color="status-error-05">
              {item.error_message || item.judge_error || "Unknown error"}
            </Text>
          </div>
        )}
        <section>
          <div className="mb-2">
            <Text as="h3" font="main-ui-action" color="text-04">
              Model answer
            </Text>
          </div>
          <div className="rounded-lg border border-border-01 bg-background-01 p-4">
            {item.final_result ? (
              <MinimalMarkdown content={item.final_result} />
            ) : (
              <Text font="main-ui-body" color="text-02">
                No answer captured.
              </Text>
            )}
          </div>
        </section>
        <section>
          <div className="mb-2">
            <Text as="h3" font="main-ui-action" color="text-04">
              Citation performance
            </Text>
          </div>
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
                <Text font="main-ui-action" color="text-04">
                  {`[${String(source.citation_number)}] ${String(
                    source.file_name ?? source.semantic_identifier ?? "Source"
                  )}`}
                </Text>
                <div className="mt-1">
                  <Text font="secondary-body" color="text-02">
                    {Array.isArray(source.heading_path)
                      ? source.heading_path.join(" › ")
                      : ""}
                  </Text>
                </div>
                <div className="mt-2">
                  <Text as="p" font="main-ui-body" color="text-03" maxLines={5}>
                    {String(source.excerpt ?? "")}
                  </Text>
                </div>
                <div className="mt-2 break-all">
                  <Text font="secondary-body" color="text-02">
                    {String(
                      source.regulatory_chunk_id ?? "No regulatory chunk id"
                    )}
                  </Text>
                </div>
              </div>
            ))}
          </div>
        </section>
        <section>
          <div className="mb-2">
            <Text as="h3" font="main-ui-action" color="text-04">
              Production chat execution steps
            </Text>
          </div>
          <div className="flex flex-col gap-2 border-l-2 border-border-02 pl-4">
            {item.execution_steps.map((step, index) => (
              <details
                key={index}
                className="rounded-lg border border-border-01 p-3"
              >
                <summary className="cursor-pointer">
                  <Text font="main-ui-action" color="text-04">
                    {`${index + 1}. ${String(step.title ?? step.kind ?? "Step")}`}
                  </Text>
                </summary>
                <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-background-01 p-3">
                  <Text font="secondary-mono" color="text-03">
                    {JSON.stringify(step, null, 2)}
                  </Text>
                </pre>
              </details>
            ))}
          </div>
        </section>
        <section>
          <div className="mb-2">
            <Text as="h3" font="main-ui-action" color="text-04">
              LLM calls and usage
            </Text>
          </div>
          <div className="overflow-x-auto rounded-lg border border-border-01">
            <table className="w-full text-left text-sm">
              <thead className="bg-background-01 text-text-02">
                <tr>
                  {(
                    ["Phase", "Model", "Input", "Output", "Cache"] as const
                  ).map((label) => (
                    <th key={label} className="p-2">
                      <Text font="secondary-action" color="text-02">
                        {label}
                      </Text>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {item.llm_calls.map((call, index) => (
                  <tr key={index} className="border-t border-border-01">
                    {[
                      call.phase ?? "answer",
                      call.model ?? "—",
                      call.input_tokens ?? 0,
                      call.output_tokens ?? 0,
                      call.cache_read_tokens ?? 0,
                    ].map((value, valueIndex) => (
                      <td key={valueIndex} className="p-2">
                        <Text font="main-ui-body" color="text-03">
                          {String(value)}
                        </Text>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-2">
            <Text font="secondary-body" color="text-02">
              {`Answer cost: ${formatCost(item.cost_cents)} · Judge cost: ${formatCost(item.judgment?.cost_cents ?? null)} · Chat session: ${item.chat_session_id ?? "—"}`}
            </Text>
          </div>
        </section>
        {item.answer_reasoning && (
          <details>
            <summary className="cursor-pointer">
              <Text font="main-ui-action" color="text-04">
                Model-provided reasoning
              </Text>
            </summary>
            <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-border-01 bg-background-01 p-4">
              <Text font="main-ui-mono" color="text-03">
                {item.answer_reasoning}
              </Text>
            </pre>
          </details>
        )}
        <section>
          <div className="mb-2">
            <Text as="h3" font="main-ui-action" color="text-04">
              Judge report
            </Text>
          </div>
          <JudgeReport item={item} />
        </section>
      </div>
    </details>
  );
}
