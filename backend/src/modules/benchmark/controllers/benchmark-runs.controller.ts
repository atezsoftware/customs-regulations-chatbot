import {inject} from '@loopback/core';
import {repository} from '@loopback/repository';
import {get, HttpErrors, param, post, Request, requestBody, response, RestBindings} from '@loopback/rest';
import {getCurrentUser, requireAdmin} from '../../../common/auth';
import {UserRepository} from '../../auth/repositories';
import {LlmModelRepository} from '../../llm-catalog/repositories';
import {BenchmarkRun} from '../models';
import {
  BenchmarkModelMetricsRow,
  BenchmarkModelPercentilesRow,
  BenchmarkQuestionRepository,
  BenchmarkRunItemRepository,
  BenchmarkRunJudgmentRepository,
  BenchmarkRunRepository,
} from '../repositories';

interface CreateRunBody {
  label?: string;
  providerModelPairs: Array<{provider: string; modelId: string}>;
  questionIds: number[] | 'all-active';
  judgeProvider: string;
  judgeModel: string;
}

function toNumber(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toSafeRun(run: BenchmarkRun) {
  return {
    id: run.id,
    label: run.label ?? null,
    status: run.status,
    judgeProvider: run.judgeProvider,
    judgeModel: run.judgeModel,
    totalItems: run.totalItems,
    completedItems: run.completedItems,
    failedItems: run.failedItems,
    startedAt: run.startedAt ?? null,
    completedAt: run.completedAt ?? null,
    createdAt: run.createdAt,
  };
}

function toSafeMetrics(row: BenchmarkModelMetricsRow, percentiles?: BenchmarkModelPercentilesRow) {
  return {
    provider: row.provider,
    modelId: row.model_id,
    totalCount: toNumber(row.total_count) ?? 0,
    completedCount: toNumber(row.completed_count) ?? 0,
    errorCount: toNumber(row.error_count) ?? 0,
    avgSteps: toNumber(row.avg_steps),
    avgTokensPerStep: toNumber(row.avg_tokens_per_step),
    avgTotalTokens: toNumber(row.avg_total_tokens),
    avgDurationMs: toNumber(row.avg_duration_ms),
    avgDurationPerStepMs: toNumber(row.avg_duration_per_step_ms),
    p50DurationMs: toNumber(percentiles?.p50_duration_ms),
    p95DurationMs: toNumber(percentiles?.p95_duration_ms),
    avgCostUsd: toNumber(row.avg_cost_usd),
    totalCostUsd: toNumber(row.total_cost_usd),
    successRate: toNumber(row.success_rate),
    errorRate: toNumber(row.error_rate),
    citationRate: toNumber(row.citation_rate),
    avgApiCalls: toNumber(row.avg_api_calls),
    avgContextSummaries: toNumber(row.avg_context_summaries),
    judgeOverallScore: toNumber(row.judge_overall_score),
    judgeCorrectness: toNumber(row.judge_correctness),
    judgeGroundedness: toNumber(row.judge_groundedness),
    judgeCompleteness: toNumber(row.judge_completeness),
    judgeClarity: toNumber(row.judge_clarity),
  };
}

export class BenchmarkRunsController {
  constructor(
    @repository(BenchmarkRunRepository) private runs: BenchmarkRunRepository,
    @repository(BenchmarkRunItemRepository) private items: BenchmarkRunItemRepository,
    @repository(BenchmarkQuestionRepository) private questions: BenchmarkQuestionRepository,
    @repository(BenchmarkRunJudgmentRepository) private judgments: BenchmarkRunJudgmentRepository,
    @repository(LlmModelRepository) private llmModels: LlmModelRepository,
    @repository(UserRepository) private userRepository: UserRepository,
    @inject(RestBindings.Http.REQUEST) private request: Request,
  ) {}

  @post('/admin/benchmark/runs')
  @response(202, {description: 'Started a benchmark run'})
  async create(@requestBody() body: CreateRunBody) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    if (!body.providerModelPairs?.length) {
      throw new HttpErrors.BadRequest('Select at least one model.');
    }
    for (const pair of body.providerModelPairs) {
      const model = await this.llmModels.findOne({
        where: {provider: pair.provider, modelId: pair.modelId, isActive: true},
      });
      if (!model) throw new HttpErrors.BadRequest(`Model not available: ${pair.provider}/${pair.modelId}`);
    }
    const judge = await this.llmModels.findOne({
      where: {provider: body.judgeProvider, modelId: body.judgeModel, isActive: true},
    });
    if (!judge) throw new HttpErrors.BadRequest('Judge model is not available.');

    const questionIds =
      body.questionIds === 'all-active'
        ? (await this.questions.activeQuestions()).map(question => question.id!)
        : body.questionIds;
    if (!questionIds?.length) {
      throw new HttpErrors.BadRequest('Select at least one question.');
    }
    const foundQuestions = await this.questions.find({where: {id: {inq: questionIds}}});
    if (foundQuestions.length !== questionIds.length) {
      throw new HttpErrors.BadRequest('One or more selected questions do not exist.');
    }

    const totalItems = body.providerModelPairs.length * questionIds.length;
    const run = await this.runs.create({
      label: body.label?.trim() || undefined,
      status: 'running',
      judgeProvider: body.judgeProvider,
      judgeModel: body.judgeModel,
      createdBy: user.id,
      totalItems,
      completedItems: 0,
      failedItems: 0,
      startedAt: new Date().toISOString(),
    });

    for (const pair of body.providerModelPairs) {
      for (const questionId of questionIds) {
        await this.items.create({
          runId: run.id!,
          provider: pair.provider,
          modelId: pair.modelId,
          questionId,
          repeatIndex: 1,
          status: 'pending',
          incomplete: false,
        });
      }
    }

    return {runId: run.id};
  }

  @get('/admin/benchmark/runs')
  @response(200, {description: 'List benchmark runs'})
  async list() {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const runs = await this.runs.find({order: ['id DESC'], limit: 100});
    return {runs: runs.map(toSafeRun)};
  }

  @get('/admin/benchmark/runs/{id}')
  @response(200, {description: 'Benchmark run status and per-model metrics'})
  async get(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const run = await this.runs.findById(id).catch(() => null);
    if (!run) throw new HttpErrors.NotFound('Benchmark run not found.');

    const [metrics, percentiles] = await Promise.all([
      this.items.aggregateByModel(id),
      this.items.percentilesByModel(id),
    ]);
    const percentilesByKey = new Map(
      percentiles.map(row => [`${row.provider}/${row.model_id}`, row]),
    );

    return {
      run: toSafeRun(run),
      metrics: metrics.map(row => toSafeMetrics(row, percentilesByKey.get(`${row.provider}/${row.model_id}`))),
    };
  }

  @get('/admin/benchmark/runs/{id}/items')
  @response(200, {description: 'Per-question benchmark run items'})
  async listItems(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const run = await this.runs.findById(id).catch(() => null);
    if (!run) throw new HttpErrors.NotFound('Benchmark run not found.');

    const rows = await this.items.find({where: {runId: id}, order: ['id ASC']});
    const itemIds = rows.map(row => row.id!).filter(Boolean);
    const judgmentRows = itemIds.length
      ? await this.judgments.find({where: {runItemId: {inq: itemIds}}})
      : [];
    const judgmentByItemId = new Map(judgmentRows.map(judgment => [judgment.runItemId, judgment]));

    return {
      items: rows.map(row => {
        const judgment = judgmentByItemId.get(row.id!);
        return {
          id: row.id,
          provider: row.provider,
          modelId: row.modelId,
          questionId: row.questionId,
          status: row.status,
          incomplete: row.incomplete,
          errorMessage: row.errorMessage ?? null,
          finalResult: row.finalResult ?? null,
          citedSources: row.citedSources ?? [],
          stepPath: row.stepPath ?? [],
          steps: row.steps ?? null,
          totalTokens: row.totalTokens ?? null,
          durationMs: row.durationMs ?? null,
          costUsd: row.costUsd ?? null,
          startedAt: row.startedAt ?? null,
          completedAt: row.completedAt ?? null,
          judgment: judgment
            ? {
                overallScore: judgment.overallScore,
                correctnessScore: judgment.correctnessScore,
                groundednessScore: judgment.groundednessScore,
                completenessScore: judgment.completenessScore,
                clarityScore: judgment.clarityScore,
                rationale: judgment.rationale ?? null,
              }
            : null,
        };
      }),
    };
  }

  @post('/admin/benchmark/runs/{id}/cancel')
  @response(204, {description: 'Cancelled a benchmark run'})
  async cancel(@param.path.number('id') id: number) {
    const user = await getCurrentUser(this.request, this.userRepository);
    requireAdmin(user);

    const run = await this.runs.findById(id).catch(() => null);
    if (!run) throw new HttpErrors.NotFound('Benchmark run not found.');
    if (run.status !== 'pending' && run.status !== 'running') {
      throw new HttpErrors.BadRequest('Only a pending or running benchmark run can be cancelled.');
    }
    // Items already `running` may still finish their in-flight call; no
    // new `pending` item is claimed once the run leaves `running` status
    // (see BenchmarkRunnerService.claimPendingItems's join on run status).
    await this.runs.updateById(id, {status: 'cancelled', completedAt: new Date().toISOString()});
  }
}
