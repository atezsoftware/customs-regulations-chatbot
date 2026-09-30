import {
  buildGraphLayout,
  buildGraphPresentation,
  type GraphNode,
} from "../graphPresentation";

function node(overrides: Partial<GraphNode>): GraphNode {
  return {
    node_id: "root",
    parent_node_id: null,
    kind: "step",
    operation: "chat.input",
    status: "COMPLETE",
    capture_status: "COMPLETE",
    started_at: "2026-09-30T06:00:00Z",
    ended_at: "2026-09-30T06:00:01Z",
    attributes: {},
    has_input: false,
    has_output: false,
    has_reasoning: false,
    error: null,
    ...overrides,
  };
}

it("keeps repeated calls distinct and assigns stable chronology, owner and relations", () => {
  const root = node({ attributes: { agent: "run_llm_loop" } });
  const first = node({
    node_id: "first",
    parent_node_id: "root",
    kind: "generation",
    operation: "regulatory_coverage_plan",
    started_at: "2026-09-30T06:00:02Z",
  });
  const provider = node({
    node_id: "provider",
    parent_node_id: "first",
    operation: "llm.provider_attempt",
    started_at: "2026-09-30T06:00:03Z",
  });
  const retry = node({
    node_id: "retry",
    parent_node_id: "root",
    kind: "generation",
    operation: "regulatory_coverage_plan",
    started_at: "2026-09-30T06:00:04Z",
  });
  const presented = buildGraphPresentation(
    [retry, provider, root, first],
    [{ from_node_id: "first", to_node_id: "retry", kind: "data" }]
  );

  expect(presented.map((item) => item.node.node_id)).toEqual([
    "root",
    "first",
    "provider",
    "retry",
  ]);
  expect(presented.map((item) => item.sequence)).toEqual([1, 2, 3, 4]);
  expect(presented[2]).toMatchObject({
    parentSequence: 2,
    depth: 2,
    agent: "run_llm_loop",
    phase: "model",
  });
  expect(presented[3]).toMatchObject({
    parentSequence: 1,
    dataInputSequences: [2],
    callNumber: 2,
    phase: "planning",
  });
});

it("keeps concurrent research agents in separate lanes at the same dependency stage", () => {
  const root = node({ attributes: { agent: "Chat agent" } });
  const agentOne = node({
    node_id: "agent-one",
    parent_node_id: "root",
    kind: "function",
    operation: "research_agent",
    attributes: { agent: "Research agent 1" },
    started_at: "2026-09-30T06:00:02Z",
    ended_at: "2026-09-30T06:00:08Z",
  });
  const agentTwo = node({
    node_id: "agent-two",
    parent_node_id: "root",
    kind: "function",
    operation: "research_agent",
    attributes: { agent: "Research agent 2" },
    started_at: "2026-09-30T06:00:02Z",
    ended_at: "2026-09-30T06:00:07Z",
  });
  const search = node({
    node_id: "search",
    parent_node_id: "agent-one",
    operation: "bm25.search",
    started_at: "2026-09-30T06:00:03Z",
    ended_at: "2026-09-30T06:00:04Z",
  });
  const presented = buildGraphPresentation(
    [search, agentTwo, root, agentOne],
    []
  );
  const layout = buildGraphLayout(presented, []);
  const byId = new Map(layout.nodes.map((item) => [item.node.node_id, item]));

  expect(byId.get("agent-one")?.rank).toBe(byId.get("agent-two")?.rank);
  expect(byId.get("agent-one")?.y).not.toBe(byId.get("agent-two")?.y);
  expect(byId.get("search")?.rank).toBeGreaterThan(byId.get("agent-one")!.rank);
  expect(byId.get("search")?.agent).toBe("Research agent 1");
  expect(layout.lanes.map((lane) => lane.agent)).toEqual([
    "Chat agent",
    "Research agent 1",
    "Research agent 2",
  ]);
});
