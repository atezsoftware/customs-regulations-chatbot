import {PostgresDataSource} from '../../../datasources';
import {resolveCoreApiRestUrl, resolveDatabaseUrl, virtualCorpusKey} from '../../directories/services';
import {BenchmarkRunItem} from '../models';
import {
  BenchmarkQuestionDirectoryRepository,
  BenchmarkQuestionRepository,
  BenchmarkRunItemRepository,
  BenchmarkRunJudgmentRepository,
  BenchmarkRunRepository,
} from '../repositories';

// Distinct from OpenRouter catalog sync's advisory lock id (7412109) so the
// two background jobs never contend for the same lock.
const ADVISORY_LOCK_ID = 7412110;
const DEFAULT_MAX_CONCURRENCY = 3;
const DEFAULT_STALE_MINUTES = 10;

interface RunQuestionStats {
  steps: number;
  api_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  thinking_tokens: number;
  total_tokens: number;
  tool_result_chars: number;
  context_summaries: number;
  duration_ms: number;
  cost_usd: string | null;
  cost_source: 'provider' | 'estimated' | null;
}

interface RunQuestionResult {
  final_result: string;
  error: string | null;
  incomplete: boolean;
  cited_sources: string[];
  step_path: string[];
  stats: RunQuestionStats;
}

interface JudgeResult {
  correctness: number;
  groundedness: number;
  completeness: number;
  clarity: number;
  overall_score: number;
  rationale: string;
}

/**
 * Drives pending benchmark items to completion in the background.
 *
 * Mirrors `OpenRouterCatalogService`'s shape on purpose: a Postgres advisory
 * lock so only one backend replica works a tick at a time, called from a
 * `setInterval` in `index.ts`. Each item is one (model, question) pair —
 * `callRunQuestion` drives the candidate model through core-api's headless
 * benchmark runner, `callJudge` scores the resulting answer, and the run's
 * progress counters are updated atomically so concurrent items never race.
 */
export class BenchmarkRunnerService {
  constructor(
    private readonly dataSource: PostgresDataSource,
    private readonly runs: BenchmarkRunRepository,
    private readonly items: BenchmarkRunItemRepository,
    private readonly judgments: BenchmarkRunJudgmentRepository,
    private readonly questions: BenchmarkQuestionRepository,
    private readonly questionDirectories: BenchmarkQuestionDirectoryRepository,
  ) {}

  async tick(): Promise<void> {
    const staleMinutes = Math.max(
      1,
      Number(process.env.BENCHMARK_STALE_ITEM_MINUTES ?? DEFAULT_STALE_MINUTES),
    );
    await this.items.reclaimStaleItems(staleMinutes);

    const lock = (await this.dataSource.execute(`SELECT pg_try_advisory_lock($1) AS locked`, [
      ADVISORY_LOCK_ID,
    ])) as Array<{locked: boolean}>;
    if (!lock[0]?.locked) return;

    try {
      const maxConcurrency = Math.max(
        1,
        Number(process.env.BENCHMARK_MAX_CONCURRENCY ?? DEFAULT_MAX_CONCURRENCY),
      );
      const claimed = await this.items.claimPendingItems(maxConcurrency);
      if (!claimed.length) return;
      await Promise.all(claimed.map(item => this.executeItem(item)));
    } finally {
      await this.dataSource.execute(`SELECT pg_advisory_unlock($1)`, [ADVISORY_LOCK_ID]);
    }
  }

  private async executeItem(item: BenchmarkRunItem): Promise<void> {
    try {
      const [question, run, links] = await Promise.all([
        this.questions.findById(item.questionId),
        this.runs.findById(item.runId),
        this.questionDirectories.find({where: {questionId: item.questionId}}),
      ]);
      const indexFolders = links.map(link => virtualCorpusKey(link.directoryId));
      if (!indexFolders.length) {
        throw new Error('Benchmark question has no linked directories.');
      }

      const runResult = await this.callRunQuestion({
        task: question.prompt,
        indexFolders,
        provider: item.provider,
        model: item.modelId,
      });

      if (runResult.error) {
        await this.markItemError(item, runResult.error);
        return;
      }

      let judgment: JudgeResult | null = null;
      try {
        judgment = await this.callJudge({
          question: question.prompt,
          referenceAnswer: question.referenceAnswer,
          expectedFacts: question.expectedFacts,
          rubricNotes: question.rubricNotes,
          candidateAnswer: runResult.final_result,
          citedSources: runResult.cited_sources,
          judgeProvider: run.judgeProvider,
          judgeModel: run.judgeModel,
        });
      } catch (error) {
        // Judging is best-effort: a candidate run that itself succeeded
        // should still count as completed even if the judge call fails
        // (rate limit, judge model outage) — metrics stay valid, just
        // without a score for this item.
        console.warn(
          `[benchmark] judge call failed item=${item.id}:`,
          error instanceof Error ? error.message : error,
        );
      }

      await this.items.updateById(item.id, {
        status: 'completed',
        finalResult: runResult.final_result,
        incomplete: runResult.incomplete,
        steps: runResult.stats.steps,
        apiCalls: runResult.stats.api_calls,
        promptTokens: runResult.stats.prompt_tokens,
        completionTokens: runResult.stats.completion_tokens,
        thinkingTokens: runResult.stats.thinking_tokens,
        totalTokens: runResult.stats.total_tokens,
        toolResultChars: runResult.stats.tool_result_chars,
        contextSummaries: runResult.stats.context_summaries,
        durationMs: runResult.stats.duration_ms,
        costUsd: runResult.stats.cost_usd ?? undefined,
        costSource: runResult.stats.cost_source ?? undefined,
        citedSources: runResult.cited_sources,
        stepPath: runResult.step_path,
        completedAt: new Date().toISOString(),
      });

      if (judgment) {
        await this.judgments.create({
          runItemId: item.id!,
          judgeProvider: run.judgeProvider,
          judgeModel: run.judgeModel,
          correctnessScore: judgment.correctness,
          groundednessScore: judgment.groundedness,
          completenessScore: judgment.completeness,
          clarityScore: judgment.clarity,
          overallScore: judgment.overall_score,
          rationale: judgment.rationale,
        });
      }

      await this.runs.recordItemOutcome(item.runId, 'completed');
    } catch (error) {
      await this.markItemError(
        item,
        error instanceof Error ? error.message : 'Benchmark item failed.',
      );
    }
  }

  private async markItemError(item: BenchmarkRunItem, message: string): Promise<void> {
    await this.items.updateById(item.id, {
      status: 'error',
      errorMessage: message.slice(0, 2000),
      completedAt: new Date().toISOString(),
    });
    await this.runs.recordItemOutcome(item.runId, 'failed');
  }

  private async callRunQuestion(input: {
    task: string;
    indexFolders: string[];
    provider: string;
    model: string;
  }): Promise<RunQuestionResult> {
    const endpoint = `${resolveCoreApiRestUrl()}/api/benchmark/run-question`;
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: this.internalHeaders(),
      body: JSON.stringify({
        task: input.task,
        index_folders: input.indexFolders,
        database_url: resolveDatabaseUrl() ?? null,
        provider: input.provider,
        model: input.model,
      }),
    });
    const body = (await res.json()) as Record<string, unknown>;
    if (!res.ok) {
      throw new Error(
        typeof body.error === 'string' ? body.error : `Benchmark run-question failed (${res.status}).`,
      );
    }
    return body as unknown as RunQuestionResult;
  }

  private async callJudge(input: {
    question: string;
    referenceAnswer?: string;
    expectedFacts?: string[];
    rubricNotes?: string;
    candidateAnswer: string;
    citedSources: string[];
    judgeProvider: string;
    judgeModel: string;
  }): Promise<JudgeResult> {
    const endpoint = `${resolveCoreApiRestUrl()}/api/benchmark/judge`;
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: this.internalHeaders(),
      body: JSON.stringify({
        question: input.question,
        reference_answer: input.referenceAnswer ?? null,
        expected_facts: input.expectedFacts ?? null,
        rubric_notes: input.rubricNotes ?? null,
        candidate_answer: input.candidateAnswer,
        cited_sources: input.citedSources,
        judge_provider: input.judgeProvider,
        judge_model: input.judgeModel,
      }),
    });
    const body = (await res.json()) as Record<string, unknown>;
    if (!res.ok) {
      throw new Error(
        typeof body.error === 'string' ? body.error : `Benchmark judge failed (${res.status}).`,
      );
    }
    return body as unknown as JudgeResult;
  }

  private internalHeaders(): Record<string, string> {
    const headers: Record<string, string> = {'Content-Type': 'application/json'};
    if (process.env.CORE_INTERNAL_TOKEN) headers['X-Internal-Token'] = process.env.CORE_INTERNAL_TOKEN;
    return headers;
  }
}
