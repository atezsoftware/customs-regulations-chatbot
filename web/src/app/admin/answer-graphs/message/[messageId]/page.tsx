"use client";

import { use, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { Button, InputTypeIn } from "@opal/components";
import { useUser } from "@/providers/UserProvider";

type Run = {
  run_id: string | null;
  status: string;
  capture_status: string | null;
  capture_error: string | null;
  model_name: string | null;
};
type Node = {
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
type Edge = { from_node_id: string; to_node_id: string; kind: string };
type Page<T> = T & { next_offset: number | null };
type Detail = {
  node: Node;
  input: unknown;
  output: unknown;
  reasoning: unknown;
  input_state: string;
  output_state: string;
  reasoning_state: string;
};

const PAGE_SIZE = 200;
const NODE_WIDTH = 220;
const NODE_HEIGHT = 76;

async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(
      response.status === 403
        ? "You do not have permission to view this graph."
        : `Graph request failed (${response.status}).`
    );
  }
  return (await response.json()) as T;
}

function useGraphPages(runId: string | null, phase: string) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [nodeOffset, setNodeOffset] = useState<number | null>(0);
  const [edgeOffset, setEdgeOffset] = useState<number | null>(0);
  const [pageRequest, setPageRequest] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setNodes([]);
    setEdges([]);
    setNodeOffset(0);
    setEdgeOffset(0);
    setError(null);
    setPageRequest((request) => request + 1);
  }, [runId, phase]);

  useEffect(() => {
    if (phase !== "RUNNING" && phase !== "FINALIZING") return;
    const interval = window.setInterval(() => {
      setNodeOffset(0);
      setEdgeOffset(0);
      setPageRequest((request) => request + 1);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [phase]);

  useEffect(() => {
    if (!runId || (nodeOffset === null && edgeOffset === null)) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    const base = `/api/admin/answer-graphs/${runId}`;
    Promise.all([
      nodeOffset === null
        ? Promise.resolve(null)
        : loadJson<Page<{ nodes: Node[] }>>(
            `${base}/nodes?offset=${nodeOffset}&limit=${PAGE_SIZE}`
          ),
      edgeOffset === null
        ? Promise.resolve(null)
        : loadJson<Page<{ edges: Edge[] }>>(
            `${base}/edges?offset=${edgeOffset}&limit=${PAGE_SIZE}`
          ),
    ])
      .then(([nodePage, edgePage]) => {
        if (cancelled) return;
        if (nodePage) {
          setNodes((previous) => {
            const byId = new Map(previous.map((node) => [node.node_id, node]));
            nodePage.nodes.forEach((node) => byId.set(node.node_id, node));
            return Array.from(byId.values());
          });
          setNodeOffset(nodePage.next_offset);
        }
        if (edgePage) {
          setEdges((previous) => {
            const byId = new Map(
              previous.map((edge) => [
                `${edge.kind}:${edge.from_node_id}:${edge.to_node_id}`,
                edge,
              ])
            );
            edgePage.edges.forEach((edge) =>
              byId.set(
                `${edge.kind}:${edge.from_node_id}:${edge.to_node_id}`,
                edge
              )
            );
            return Array.from(byId.values());
          });
          setEdgeOffset(edgePage.next_offset);
        }
      })
      .catch((failure: unknown) => {
        if (!cancelled)
          setError(
            failure instanceof Error ? failure.message : "Could not load graph."
          );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, pageRequest]);

  return {
    nodes,
    edges,
    nodeOffset,
    edgeOffset,
    loading,
    error,
    loadMore: () => setPageRequest((request) => request + 1),
  };
}

function graphLayout(nodes: Node[], edges: Edge[]) {
  const byId = new Map(nodes.map((node) => [node.node_id, node]));
  const parents = new Map<string, string[]>();
  for (const node of nodes) {
    if (node.parent_node_id && byId.has(node.parent_node_id))
      parents.set(node.node_id, [node.parent_node_id]);
  }
  for (const edge of edges) {
    if (!byId.has(edge.from_node_id) || !byId.has(edge.to_node_id)) continue;
    parents.set(edge.to_node_id, [
      ...(parents.get(edge.to_node_id) || []),
      edge.from_node_id,
    ]);
  }
  const depths = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (id: string): number => {
    if (depths.has(id)) return depths.get(id)!;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const depth = Math.min(
      20,
      Math.max(
        0,
        ...(parents.get(id) || []).map((parent) => depthOf(parent) + 1)
      )
    );
    visiting.delete(id);
    depths.set(id, depth);
    return depth;
  };
  nodes.forEach((node) => depthOf(node.node_id));
  const rows = new Map<number, number>();
  const positions = new Map<string, { x: number; y: number }>();
  for (const node of nodes) {
    const depth = depths.get(node.node_id) || 0;
    const row = rows.get(depth) || 0;
    positions.set(node.node_id, {
      x: 36 + depth * (NODE_WIDTH + 100),
      y: 36 + row * (NODE_HEIGHT + 32),
    });
    rows.set(depth, row + 1);
  }
  let maxDepth = 0;
  depths.forEach((depth) => {
    maxDepth = Math.max(maxDepth, depth);
  });
  return {
    positions,
    width: 72 + (maxDepth + 1) * (NODE_WIDTH + 100),
    height:
      72 + Math.max(1, ...Array.from(rows.values())) * (NODE_HEIGHT + 32),
  };
}

function Payload({
  label,
  value,
  state,
}: {
  label: string;
  value: unknown;
  state: string;
}) {
  return (
    <section className="mt-4">
      <h3 className="text-sm font-semibold">
        {label}{" "}
        <span className="text-xs font-normal opacity-65">({state})</span>
      </h3>
      {state === "CAPTURED" ? (
        <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-all rounded-lg border border-border-01 bg-background-neutral-01 p-3 text-xs">
          {JSON.stringify(value, null, 2)}
        </pre>
      ) : (
        <p className="mt-1 text-sm opacity-70">
          {state === "UNAVAILABLE"
            ? "This payload could not be read."
            : state === "EXPIRED"
              ? "This payload has passed its retention period."
              : "The provider or operation did not return this field."}
        </p>
      )}
    </section>
  );
}

export default function AnswerGraphPage({
  params,
}: {
  params: Promise<{ messageId: string }>;
}) {
  const { messageId } = use(params);
  const { isAdmin, user } = useUser();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [search, setSearch] = useState("");
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({
    left: 0,
    top: 0,
    width: 1200,
    height: 800,
  });
  const graphUrl =
    isAdmin && /^\d+$/.test(messageId)
      ? `/api/admin/answer-graphs/by-message/${messageId}`
      : null;
  const {
    data: run,
    error: runError,
    isLoading: runLoading,
  } = useSWR<Run>(graphUrl, loadJson, {
    refreshInterval: (latest) =>
      latest?.status === "RUNNING" || latest?.status === "FINALIZING"
        ? 2500
        : 0,
  });
  const runId = run?.run_id || null;
  const { nodes, edges, nodeOffset, edgeOffset, loading, error, loadMore } =
    useGraphPages(runId, run?.status || "");
  const detailUrl =
    runId && selectedId
      ? `/api/admin/answer-graphs/${runId}/nodes/${encodeURIComponent(selectedId)}`
      : null;
  const { data: detail, error: detailError } = useSWR<Detail>(
    detailUrl,
    loadJson
  );
  const layout = useMemo(() => graphLayout(nodes, edges), [nodes, edges]);
  useEffect(() => {
    const element = graphContainerRef.current;
    if (!element) return;
    const update = () =>
      setViewport({
        left: element.scrollLeft / zoom,
        top: element.scrollTop / zoom,
        width: element.clientWidth / zoom,
        height: element.clientHeight / zoom,
      });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [runId, zoom]);
  const inViewport = (position: { x: number; y: number } | undefined) =>
    position !== undefined &&
    position.x + NODE_WIDTH >= viewport.left - 300 &&
    position.x <= viewport.left + viewport.width + 300 &&
    position.y + NODE_HEIGHT >= viewport.top - 300 &&
    position.y <= viewport.top + viewport.height + 300;
  const renderedNodes = nodes.filter((node) =>
    inViewport(layout.positions.get(node.node_id))
  );
  const searchResults = search
    ? nodes
        .filter((node) =>
          `${node.operation} ${node.kind} ${node.node_id}`
            .toLowerCase()
            .includes(search.toLowerCase())
        )
        .slice(0, 50)
    : [];
  const visibleEdges = useMemo(() => {
    const result = [...edges];
    const known = new Set(
      result.map((edge) => `${edge.from_node_id}:${edge.to_node_id}`)
    );
    for (const node of nodes) {
      if (
        node.parent_node_id &&
        !known.has(`${node.parent_node_id}:${node.node_id}`)
      )
        result.push({
          from_node_id: node.parent_node_id,
          to_node_id: node.node_id,
          kind: "parent",
        });
    }
    return result;
  }, [nodes, edges]);

  if (!user) return <p className="p-6">Loading account…</p>;
  if (!isAdmin) return <p className="p-6">Administrator access required.</p>;
  if (!graphUrl) return <p className="p-6">Invalid message ID.</p>;
  return (
    <main className="flex h-[calc(100vh-5rem)] min-h-[600px] flex-col p-5">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Answer execution graph</h1>
          <p className="text-sm opacity-70">Assistant message {messageId}</p>
          {run && (
            <p className="mt-1 text-sm">
              {run.status} · Capture {run.capture_status || "unknown"} ·{" "}
              {run.model_name || "model unknown"} · {nodes.length} nodes
              {nodeOffset !== null ? "+" : ""}
              {run.capture_error ? ` · ${run.capture_error}` : ""}
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <Button
            aria-label="Zoom out"
            size="sm"
            prominence="secondary"
            onClick={() => setZoom((value) => Math.max(0.5, value - 0.1))}
          >
            −
          </Button>
          <span className="self-center text-sm">{Math.round(zoom * 100)}%</span>
          <Button
            aria-label="Zoom in"
            size="sm"
            prominence="secondary"
            onClick={() => setZoom((value) => Math.min(1.5, value + 0.1))}
          >
            +
          </Button>
        </div>
      </header>
      {runLoading && <p>Loading graph…</p>}
      {runError && <p role="alert">{String(runError.message)}</p>}
      {run?.status === "NO_TRACE" && (
        <p>This answer predates graph capture or its trace was unavailable.</p>
      )}
      {error && <p role="alert">{error}</p>}
      {runId && (
        <div className="flex min-h-0 flex-1 gap-4">
          <div
            ref={graphContainerRef}
            onScroll={(event) => {
              const element = event.currentTarget;
              setViewport({
                left: element.scrollLeft / zoom,
                top: element.scrollTop / zoom,
                width: element.clientWidth / zoom,
                height: element.clientHeight / zoom,
              });
            }}
            className="min-w-0 flex-1 overflow-auto rounded-lg border border-border-01 bg-background-neutral-01"
          >
            <div
              className="relative origin-top-left"
              style={{
                width: layout.width * zoom,
                height: layout.height * zoom,
              }}
            >
              <div
                className="relative origin-top-left"
                style={{
                  width: layout.width,
                  height: layout.height,
                  transform: `scale(${zoom})`,
                }}
              >
                <svg
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0"
                  width={layout.width}
                  height={layout.height}
                >
                  {visibleEdges.map((edge, index) => {
                    const source = layout.positions.get(edge.from_node_id);
                    const target = layout.positions.get(edge.to_node_id);
                    if (!source || !target) return null;
                    if (!inViewport(source) && !inViewport(target)) return null;
                    const x1 = source.x + NODE_WIDTH;
                    const x2 = target.x;
                    const y1 = source.y + NODE_HEIGHT / 2;
                    const y2 = target.y + NODE_HEIGHT / 2;
                    return (
                      <path
                        key={`${edge.kind}:${edge.from_node_id}:${edge.to_node_id}:${index}`}
                        d={`M ${x1} ${y1} C ${x1 + 48} ${y1}, ${x2 - 48} ${y2}, ${x2} ${y2}`}
                        fill="none"
                        stroke={edge.kind === "data" ? "#4f83d1" : "#a3a3a3"}
                        strokeWidth={edge.kind === "data" ? 2 : 1}
                        strokeDasharray={
                          edge.kind === "data" ? undefined : "4 4"
                        }
                      />
                    );
                  })}
                </svg>
                {renderedNodes.map((node) => {
                  const position = layout.positions.get(node.node_id)!;
                  return (
                    <div
                      key={node.node_id}
                      className={`absolute overflow-hidden rounded-lg border bg-background-neutral-00 p-2 shadow-sm ${
                        selectedId === node.node_id
                          ? "border-blue-500"
                          : "border-border-01"
                      }`}
                      style={{
                        left: position.x,
                        top: position.y,
                        width: NODE_WIDTH,
                        height: NODE_HEIGHT,
                      }}
                    >
                      <Button
                        size="sm"
                        prominence="tertiary"
                        width="full"
                        onClick={() => setSelectedId(node.node_id)}
                        aria-label={`${node.operation}, ${node.status}`}
                        aria-pressed={selectedId === node.node_id}
                      >
                        {node.operation}
                      </Button>
                      <span className="block truncate text-xs opacity-70">
                        {node.kind} · {node.status}
                        {node.capture_status !== "COMPLETE"
                          ? ` · ${node.capture_status}`
                          : ""}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <aside className="w-[min(36rem,42%)] shrink-0 overflow-auto rounded-lg border border-border-01 p-4">
            <h2 className="font-semibold">Node details</h2>
            <label
              htmlFor="answer-graph-node-search"
              className="mt-4 block text-sm"
            >
              Find an operation
            </label>
            <InputTypeIn
              id="answer-graph-node-search"
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search operation or node ID"
            />
            {searchResults.length > 0 && (
              <ol className="mt-2 max-h-40 overflow-auto border-b border-border-01 pb-2">
                {searchResults.map((node) => (
                  <li key={node.node_id}>
                    <Button
                      size="sm"
                      prominence="tertiary"
                      width="full"
                      onClick={() => {
                        setSelectedId(node.node_id);
                        const position = layout.positions.get(node.node_id);
                        if (position && graphContainerRef.current)
                          graphContainerRef.current.scrollTo({
                            left: Math.max(0, (position.x - 80) * zoom),
                            top: Math.max(0, (position.y - 80) * zoom),
                          });
                      }}
                    >
                      {`${node.operation} · ${node.status}`}
                    </Button>
                  </li>
                ))}
              </ol>
            )}
            {!selectedId && (
              <p className="mt-2 text-sm opacity-70">
                Select a node to view its recorded input, output and reasoning.
              </p>
            )}
            {detailError && <p role="alert">{String(detailError.message)}</p>}
            {detail && detail.node.node_id === selectedId && (
              <>
                <p className="mt-2 break-all text-sm">
                  {detail.node.operation}
                </p>
                <p className="text-xs opacity-70">
                  {detail.node.started_at} · {detail.node.status}
                </p>
                {detail.node.error && (
                  <p className="mt-2 text-sm text-red-600">
                    {detail.node.error}
                  </p>
                )}
                <Payload
                  label="Input"
                  value={detail.input}
                  state={detail.input_state}
                />
                <Payload
                  label="Output"
                  value={detail.output}
                  state={detail.output_state}
                />
                <Payload
                  label="Reasoning"
                  value={detail.reasoning}
                  state={detail.reasoning_state}
                />
                <Payload
                  label="Attributes"
                  value={detail.node.attributes}
                  state="CAPTURED"
                />
              </>
            )}
          </aside>
        </div>
      )}
      {runId && (
        <footer className="mt-3 flex items-center gap-3 text-sm">
          {loading && <span>Loading operations…</span>}
          {(nodeOffset !== null || edgeOffset !== null) && !loading && (
            <Button size="sm" prominence="secondary" onClick={loadMore}>
              Load more operations
            </Button>
          )}
          <span>
            {nodes.length} nodes · {edges.length} data links
          </span>
        </footer>
      )}
    </main>
  );
}
