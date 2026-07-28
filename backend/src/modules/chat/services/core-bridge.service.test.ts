import assert from 'node:assert/strict';
import test from 'node:test';
import {CoreBridgeService, parseResearchProgress} from './core-bridge.service';

type PersistedSource = {
  messageId: number;
  title: string;
  snippet?: string;
  chunkId?: string;
  score?: number;
  id?: number;
};

type PersistedStep = {
  id?: number;
  messageId: number;
  stepKey: string;
  status: string;
  title: string;
  preview?: string;
  details?: string;
  metadata?: Record<string, unknown>;
  createdAt?: string;
  completedAt?: string;
};

function serviceWithSourceRecorder(created: PersistedSource[]) {
  const sourceRepository = {
    async create(data: PersistedSource) {
      const record = {...data, id: created.length + 1};
      created.push(record);
      return record;
    },
  };

  return new CoreBridgeService(
    undefined as never,
    undefined as never,
    undefined as never,
    undefined as never,
    sourceRepository as never,
    undefined as never,
    undefined as never,
    undefined as never,
  );
}

function serviceWithResearchSteps(steps: PersistedStep[]) {
  const researchStepRepository = {
    async find(filter: {where?: Record<string, unknown>}) {
      return steps.filter(step =>
        Object.entries(filter.where ?? {}).every(
          ([key, value]) => step[key as keyof PersistedStep] === value,
        ),
      );
    },
    async findOne(filter: {where?: Record<string, unknown>}) {
      return (
        steps.find(step =>
          Object.entries(filter.where ?? {}).every(
            ([key, value]) => step[key as keyof PersistedStep] === value,
          ),
        ) ?? null
      );
    },
    async findById(id: number) {
      const step = steps.find(candidate => candidate.id === id);
      if (!step) throw new Error(`Missing research step ${id}`);
      return step;
    },
    async updateById(id: number, data: Partial<PersistedStep>) {
      const step = steps.find(candidate => candidate.id === id);
      if (!step) throw new Error(`Missing research step ${id}`);
      Object.assign(step, data);
    },
    async create(data: PersistedStep) {
      const step = {...data, id: steps.length + 1};
      steps.push(step);
      return step;
    },
  };

  return new CoreBridgeService(
    undefined as never,
    undefined as never,
    undefined as never,
    researchStepRepository as never,
    undefined as never,
    undefined as never,
    undefined as never,
    undefined as never,
  );
}

test('multi-agent progress uses a stable event key and preserves correlation metadata', () => {
  const started = parseResearchProgress({
    event_id: 'worker-2-search-1',
    kind: 'search_started',
    sequence: 12,
    task_id: 'task-2',
    agent_id: 'worker-2',
    agent_role: 'worker',
    status: 'started',
    label: 'Searching tariff rules',
    detail: 'Looking for the applicable exception.',
  });
  const completed = parseResearchProgress({
    event_id: 'worker-2-search-1',
    kind: 'search_completed',
    sequence: 18,
    task_id: 'task-2',
    agent_id: 'worker-2',
    agent_role: 'worker',
    status: 'completed',
    label: 'Tariff search complete',
  });

  assert.equal(started?.stepKey, 'agent-worker-2-search-1');
  assert.equal(completed?.stepKey, started?.stepKey);
  assert.equal(started?.status, 'running');
  assert.equal(completed?.status, 'completed');
  assert.deepEqual(started?.metadata, {
    eventId: 'worker-2-search-1',
    kind: 'search_started',
    sequence: 12,
    taskId: 'task-2',
    agentId: 'worker-2',
    agentRole: 'worker',
  });
});

test('malformed multi-agent progress is ignored instead of creating a colliding row', () => {
  assert.equal(parseResearchProgress({status: 'started', sequence: 1}), undefined);
  assert.equal(parseResearchProgress({event_id: 'worker-1', status: 'unknown'}), undefined);
});

test('multi-agent retrieval telemetry correlates to its search progress row', async () => {
  const steps: PersistedStep[] = [
    {
      id: 1,
      messageId: 9,
      stepKey: 'agent-search-task-2-assignment-1',
      status: 'running',
      title: 'Running search',
      metadata: {
        eventId: 'search-task-2-assignment-1',
        kind: 'search_started',
        sequence: 12,
        taskId: 'task-2',
        agentId: 'worker-task-2-r1-1',
        agentRole: 'worker',
      },
    },
  ];
  const service = serviceWithResearchSteps(steps);
  const mergeRetrievalMetadata = (
    service as unknown as {
      mergeRetrievalMetadata(input: {
        messageId: number;
        data: Record<string, unknown>;
        agentStepKeysByCorrelation: Map<string, string>;
        pendingAgentRetrievalMetadata: Map<string, Record<string, unknown>>;
      }): Promise<{stepId: string; metadata?: object} | undefined>;
    }
  ).mergeRetrievalMetadata.bind(service);

  const updated = await mergeRetrievalMetadata({
    messageId: 9,
    data: {
      step: 12,
      tool_name: 'semantic_search',
      chunk_count: 4,
      chars: 1600,
      estimated_tokens: 400,
      task_id: 'task-2',
      agent_id: 'worker-task-2-r1-1',
      sequence: 12,
    },
    agentStepKeysByCorrelation: new Map(),
    pendingAgentRetrievalMetadata: new Map(),
  });

  assert.equal(updated?.stepId, 'agent-search-task-2-assignment-1');
  assert.deepEqual(steps[0].metadata, {
    eventId: 'search-task-2-assignment-1',
    kind: 'search_started',
    sequence: 12,
    taskId: 'task-2',
    agentId: 'worker-task-2-r1-1',
    agentRole: 'worker',
    chunkCount: 4,
    retrievalChars: 1600,
    retrievalTokensEstimated: 400,
    retrievalToolName: 'semantic_search',
    retrievalSequence: 12,
  });

  const saveStep = (
    service as unknown as {
      saveStep(input: {
        messageId: number;
        stepKey: string;
        status: 'completed';
        title: string;
        metadata: Record<string, unknown>;
      }): Promise<unknown>;
    }
  ).saveStep.bind(service);
  await saveStep({
    messageId: 9,
    stepKey: 'agent-search-task-2-assignment-1',
    status: 'completed',
    title: 'Search complete',
    metadata: {
      eventId: 'search-task-2-assignment-1',
      kind: 'search_completed',
      sequence: 13,
      taskId: 'task-2',
      agentId: 'worker-task-2-r1-1',
      agentRole: 'worker',
    },
  });
  assert.equal(steps[0].metadata?.chunkCount, 4);
  assert.equal(steps[0].metadata?.kind, 'search_completed');
});

test('terminal recovery closes stale agent rows from a previous socket only', async () => {
  const steps: PersistedStep[] = [
    {
      id: 1,
      messageId: 11,
      stepKey: 'agent-stale-worker',
      status: 'running',
      title: 'Searching',
      metadata: {agentId: 'stale-worker'},
    },
    {
      id: 2,
      messageId: 11,
      stepKey: 'tool-3',
      status: 'running',
      title: 'Legacy tool',
      metadata: {},
    },
    {
      id: 3,
      messageId: 11,
      stepKey: 'agent-failed-worker',
      status: 'error',
      title: 'Failed worker',
      metadata: {},
    },
  ];
  const service = serviceWithResearchSteps(steps);
  const completeRunningAgentSteps = (
    service as unknown as {
      completeRunningAgentSteps(
        messageId: number,
        activeStepKeys: ReadonlySet<string>,
      ): Promise<Array<{stepId: string; status: string}>>;
    }
  ).completeRunningAgentSteps.bind(service);

  const completed = await completeRunningAgentSteps(11, new Set());

  assert.deepEqual(
    completed.map(step => ({stepId: step.stepId, status: step.status})),
    [{stepId: 'agent-stale-worker', status: 'completed'}],
  );
  assert.equal(steps[0].status, 'completed');
  assert.ok(steps[0].completedAt);
  assert.equal(steps[1].status, 'running');
  assert.equal(steps[2].status, 'error');
});

test('exact evidence sources are persisted ahead of legacy indexed-hit fallback', async () => {
  const created: PersistedSource[] = [];
  const service = serviceWithSourceRecorder(created);
  const persistSources = (
    service as unknown as {
      persistSources(
        messageId: number,
        data: Record<string, unknown>,
        indexedHits: Array<Record<string, unknown>>,
        finalContent: string,
      ): Promise<PersistedSource[]>;
    }
  ).persistSources.bind(service);

  const result = await persistSources(
    42,
    {
      evidence_sources: [
        {
          title: 'Customs Regulation',
          locator: 'Article 5(1)',
          snippet: 'The exact evidence selected by the synthesizer.',
          chunk_id: 'exact-chunk',
          score: 0.91,
        },
      ],
      // Core's citation extractor returns the bare readable title here.
      // It must not reopen the original-query pre-search fallback after
      // an exact chunk for this title was already supplied.
      cited_sources: ['Customs Regulation'],
    },
    [
      {
        directoryId: 1,
        docId: 'legacy-doc',
        chunkId: 'wrong-presearch-chunk',
        relativePath: 'customs-regulation.pdf',
        absolutePath: '/tmp/customs-regulation.pdf',
        metadata: {article_no: '5', paragraph_no: '1'},
        text: 'A less authoritative pre-search result.',
        score: 0.7,
        citationLabel: 'Customs Regulation, Article 5(1)',
        chunkPath: 'Customs Regulation - Article 5(1)',
      },
    ],
    '[Customs Regulation, Article 5(1)]',
  );

  assert.equal(result.length, 1);
  assert.deepEqual(created, [
    {
      id: 1,
      messageId: 42,
      title: 'Customs Regulation, Article 5(1)',
      snippet: 'The exact evidence selected by the synthesizer.',
      chunkId: 'exact-chunk',
      score: 0.91,
    },
  ]);
});

test('exact evidence sources keep distinct chunks and deduplicate repeated chunk ids', async () => {
  const created: PersistedSource[] = [];
  const service = serviceWithSourceRecorder(created);
  const persistSources = (
    service as unknown as {
      persistSources(
        messageId: number,
        data: Record<string, unknown>,
        indexedHits: Array<Record<string, unknown>>,
        finalContent: string,
      ): Promise<PersistedSource[]>;
    }
  ).persistSources.bind(service);

  await persistSources(
    7,
    {
      evidence_sources: [
        {title: 'Tariff Act', locator: 'Article 1', chunk_id: 'chunk-1'},
        {title: 'Tariff Act', locator: 'Article 2', chunk_id: 'chunk-2'},
        {title: 'Duplicate label', locator: 'Article 1', chunk_id: 'chunk-1'},
      ],
    },
    [],
    '',
  );

  assert.deepEqual(
    created.map(source => ({title: source.title, chunkId: source.chunkId})),
    [
      {title: 'Tariff Act, Article 1', chunkId: 'chunk-1'},
      {title: 'Tariff Act, Article 2', chunkId: 'chunk-2'},
    ],
  );
});
