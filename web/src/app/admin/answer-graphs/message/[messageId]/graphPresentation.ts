export type GraphNode = {
  node_id: string;
  parent_node_id: string | null;
  kind: string;
  operation: string;
  status: string;
  capture_status: string;
  started_at: string;
  ended_at: string | null;
  attributes: Record<string, unknown>;
  has_input: boolean;
  has_output: boolean;
  has_reasoning: boolean;
  error: string | null;
};

export type GraphEdge = {
  from_node_id: string;
  to_node_id: string;
  kind: string;
};

export type GraphPhase =
  | "input"
  | "planning"
  | "retrieval"
  | "model"
  | "answer"
  | "operation";

export type PresentedNode = {
  node: GraphNode;
  sequence: number;
  parentSequence: number | null;
  dataInputSequences: number[];
  depth: number;
  agent: string;
  phase: GraphPhase;
  callNumber: number;
  durationMs: number | null;
};

export type PositionedNode = PresentedNode & {
  x: number;
  y: number;
  rank: number;
};
export type AgentLane = { agent: string; y: number; height: number };
export type GraphLayout = {
  nodes: PositionedNode[];
  lanes: AgentLane[];
  edges: GraphEdge[];
  width: number;
  height: number;
};

export const GRAPH_NODE_WIDTH = 240;
export const GRAPH_NODE_HEIGHT = 120;

export function graphPhase(node: GraphNode): GraphPhase {
  const operation = node.operation.toLowerCase();
  if (operation === "chat.input") return "input";
  if (operation === "answer.delivered") return "answer";
  if (node.kind === "generation" || operation.startsWith("llm.")) {
    return operation.startsWith("regulatory_") ? "planning" : "model";
  }
  if (/search|embed|rerank|retriev|bm25|keyword|vector/.test(operation)) {
    return "retrieval";
  }
  return "operation";
}

export function graphOperationLabel(node: GraphNode): string {
  if (node.operation === "chat.input") return "Question received";
  if (node.operation === "answer.delivered") return "Answer delivered";
  if (node.operation === "llm.provider_attempt") return "LLM provider call";
  return node.operation.replaceAll("_", " ").replaceAll(".", " · ");
}

export function buildGraphPresentation(
  nodes: GraphNode[],
  edges: GraphEdge[]
): PresentedNode[] {
  const ordered = [...nodes].sort(
    (left, right) =>
      Date.parse(left.started_at) - Date.parse(right.started_at) ||
      left.node_id.localeCompare(right.node_id)
  );
  const byId = new Map(ordered.map((node) => [node.node_id, node]));
  const sequenceById = new Map(
    ordered.map((node, index) => [node.node_id, index + 1])
  );
  const rootAgent = ordered.find((node) => node.operation === "chat.input")
    ?.attributes.agent;
  const fallbackAgent =
    typeof rootAgent === "string" && rootAgent
      ? rootAgent
      : "Answer workflow · agent unrecorded";
  const ownerById = new Map<string, string>();
  const depthById = new Map<string, number>();
  const resolveOwner = (node: GraphNode, seen: Set<string>): string => {
    if (ownerById.has(node.node_id)) return ownerById.get(node.node_id)!;
    if (seen.has(node.node_id)) return fallbackAgent;
    seen.add(node.node_id);
    const agentName =
      node.operation === "research_agent"
        ? node.attributes.agent
        : node.attributes.name;
    const owner =
      (node.kind === "agent" || node.operation === "research_agent") &&
      typeof agentName === "string" &&
      agentName
        ? agentName
        : node.parent_node_id && byId.has(node.parent_node_id)
          ? resolveOwner(byId.get(node.parent_node_id)!, seen)
          : fallbackAgent;
    ownerById.set(node.node_id, owner);
    seen.delete(node.node_id);
    return owner;
  };
  const resolveDepth = (node: GraphNode, seen: Set<string>): number => {
    if (depthById.has(node.node_id)) return depthById.get(node.node_id)!;
    if (seen.has(node.node_id)) return 0;
    seen.add(node.node_id);
    const depth =
      node.parent_node_id && byId.has(node.parent_node_id)
        ? Math.min(3, resolveDepth(byId.get(node.parent_node_id)!, seen) + 1)
        : 0;
    depthById.set(node.node_id, depth);
    seen.delete(node.node_id);
    return depth;
  };
  const dataInputs = new Map<string, number[]>();
  for (const edge of edges) {
    if (edge.kind !== "data") continue;
    const sourceSequence = sequenceById.get(edge.from_node_id);
    if (sourceSequence === undefined || !sequenceById.has(edge.to_node_id))
      continue;
    dataInputs.set(edge.to_node_id, [
      ...(dataInputs.get(edge.to_node_id) ?? []),
      sourceSequence,
    ]);
  }
  const calls = new Map<string, number>();
  return ordered.map((node, index) => {
    const callKey = `${node.parent_node_id ?? "root"}:${node.operation}`;
    const callNumber = (calls.get(callKey) ?? 0) + 1;
    calls.set(callKey, callNumber);
    return {
      node,
      sequence: index + 1,
      parentSequence: node.parent_node_id
        ? (sequenceById.get(node.parent_node_id) ?? null)
        : null,
      dataInputSequences: Array.from(
        new Set(dataInputs.get(node.node_id) ?? [])
      ).sort((left, right) => left - right),
      depth: resolveDepth(node, new Set()),
      agent: resolveOwner(node, new Set()),
      phase: graphPhase(node),
      callNumber,
      durationMs: node.ended_at
        ? Math.max(0, Date.parse(node.ended_at) - Date.parse(node.started_at))
        : null,
    };
  });
}

/** Dependency columns plus agent swimlanes. Time only orders unrelated siblings. */
export function buildGraphLayout(
  presented: PresentedNode[],
  edges: GraphEdge[]
): GraphLayout {
  const byId = new Map(presented.map((item) => [item.node.node_id, item]));
  const predecessors = new Map<string, Set<string>>();
  const displayedEdges: GraphEdge[] = [];
  const addPredecessor = (from: string, to: string, kind: string) => {
    if (from === to || !byId.has(from) || !byId.has(to)) return;
    if (byId.get(from)!.sequence >= byId.get(to)!.sequence) return;
    const incoming = predecessors.get(to) ?? new Set<string>();
    incoming.add(from);
    predecessors.set(to, incoming);
    displayedEdges.push({ from_node_id: from, to_node_id: to, kind });
  };
  for (const item of presented) {
    if (item.node.parent_node_id) {
      addPredecessor(item.node.parent_node_id, item.node.node_id, "parent");
    }
  }
  for (const edge of edges) {
    addPredecessor(edge.from_node_id, edge.to_node_id, edge.kind);
  }
  const previousByAgent = new Map<string, PresentedNode[]>();
  for (const item of presented) {
    const previous = previousByAgent.get(item.agent) ?? [];
    const startedAt = Date.parse(item.node.started_at);
    let latestFinished: PresentedNode | undefined;
    let latestEnd = -Infinity;
    for (const candidate of previous) {
      const endedAt = candidate.node.ended_at
        ? Date.parse(candidate.node.ended_at)
        : Infinity;
      if (endedAt <= startedAt && endedAt > latestEnd) {
        latestFinished = candidate;
        latestEnd = endedAt;
      }
    }
    if (latestFinished) {
      const incoming = predecessors.get(item.node.node_id);
      if (!incoming?.has(latestFinished.node.node_id)) {
        addPredecessor(latestFinished.node.node_id, item.node.node_id, "order");
      }
    }
    previous.push(item);
    previousByAgent.set(item.agent, previous);
  }
  const ranks = new Map<string, number>();
  for (const item of presented) {
    const incoming = predecessors.get(item.node.node_id) ?? new Set();
    const rank = Math.max(
      0,
      ...Array.from(incoming, (id) => (ranks.get(id) ?? 0) + 1)
    );
    ranks.set(item.node.node_id, rank);
  }
  const agents = Array.from(new Set(presented.map((item) => item.agent)));
  const lanes: AgentLane[] = [];
  const positioned: PositionedNode[] = [];
  let laneY = 24;
  for (const agent of agents) {
    const inLane = presented.filter((item) => item.agent === agent);
    const byRank = new Map<number, PresentedNode[]>();
    for (const item of inLane) {
      const rank = ranks.get(item.node.node_id) ?? 0;
      byRank.set(rank, [...(byRank.get(rank) ?? []), item]);
    }
    const slots = Math.max(
      1,
      ...Array.from(byRank.values(), (items) => items.length)
    );
    const height = 56 + slots * (GRAPH_NODE_HEIGHT + 22);
    lanes.push({ agent, y: laneY, height });
    byRank.forEach((items, rank) => {
      items.forEach((item, slot) => {
        positioned.push({
          ...item,
          rank,
          x: 40 + rank * (GRAPH_NODE_WIDTH + 96),
          y: laneY + 46 + slot * (GRAPH_NODE_HEIGHT + 22),
        });
      });
    });
    laneY += height + 18;
  }
  const maxRank = Math.max(0, ...Array.from(ranks.values()));
  return {
    nodes: positioned,
    lanes,
    edges: displayedEdges,
    width: 80 + (maxRank + 1) * (GRAPH_NODE_WIDTH + 96),
    height: laneY + 20,
  };
}
