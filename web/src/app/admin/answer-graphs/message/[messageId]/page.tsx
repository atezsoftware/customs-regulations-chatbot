"use client";

import { use, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { Button, InputTypeIn } from "@opal/components";
import { useUser } from "@/providers/UserProvider";
import {
  buildGraphLayout,
  buildGraphPresentation,
  GRAPH_NODE_HEIGHT,
  GRAPH_NODE_WIDTH,
  graphOperationLabel,
  type GraphEdge,
  type GraphNode,
  type GraphPhase,
} from "./graphPresentation";

type Run = {
  run_id: string | null;
  status: string;
  capture_status: string | null;
  capture_error: string | null;
  model_name: string | null;
};
type Page<T> = T & { next_offset: number | null };
type Detail = {
  node: GraphNode;
  input: unknown;
  output: unknown;
  reasoning: unknown;
  input_state: string;
  output_state: string;
  reasoning_state: string;
};

const PAGE_SIZE = 200;
const PHASE_STYLES: Record<GraphPhase, string> = {
  input: "border-slate-500 bg-slate-500/10",
  planning: "border-violet-500 bg-violet-500/10",
  retrieval: "border-cyan-500 bg-cyan-500/10",
  model: "border-blue-500 bg-blue-500/10",
  answer: "border-emerald-500 bg-emerald-500/10",
  operation: "border-amber-500 bg-amber-500/10",
};

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
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
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
        : loadJson<Page<{ nodes: GraphNode[] }>>(
            `${base}/nodes?offset=${nodeOffset}&limit=${PAGE_SIZE}`
          ),
      edgeOffset === null
        ? Promise.resolve(null)
        : loadJson<Page<{ edges: GraphEdge[] }>>(
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
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({
    left: 0,
    top: 0,
    width: 1000,
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
  const presented = useMemo(
    () => buildGraphPresentation(nodes, edges),
    [nodes, edges]
  );
  const layout = useMemo(
    () => buildGraphLayout(presented, edges),
    [presented, edges]
  );
  const positions = useMemo(
    () => new Map(layout.nodes.map((item) => [item.node.node_id, item])),
    [layout]
  );
  const presentedBySequence = useMemo(
    () => new Map(presented.map((item) => [item.sequence, item])),
    [presented]
  );
  const presentedById = useMemo(
    () => new Map(presented.map((item) => [item.node.node_id, item])),
    [presented]
  );
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
  const inViewport = (x: number, y: number) =>
    x + GRAPH_NODE_WIDTH >= viewport.left - 300 &&
    x <= viewport.left + viewport.width + 300 &&
    y + GRAPH_NODE_HEIGHT >= viewport.top - 300 &&
    y <= viewport.top + viewport.height + 300;
  const renderedNodes = layout.nodes.filter((item) =>
    inViewport(item.x, item.y)
  );
  const visibleEdges = layout.edges.filter(
    (edge) =>
      positions.has(edge.from_node_id) &&
      positions.has(edge.to_node_id) &&
      (inViewport(
        positions.get(edge.from_node_id)!.x,
        positions.get(edge.from_node_id)!.y
      ) ||
        inViewport(
          positions.get(edge.to_node_id)!.x,
          positions.get(edge.to_node_id)!.y
        ))
  );
  const searchResults = search
    ? presented
        .filter((item) =>
          `${item.node.operation} ${item.node.kind} ${item.node.node_id} ${item.agent}`
            .toLowerCase()
            .includes(search.toLowerCase())
        )
        .slice(0, 50)
    : [];
  const selected = selectedId ? presentedById.get(selectedId) : null;
  const jumpToSequence = (sequence: number) => {
    const item = presentedBySequence.get(sequence);
    if (!item) return;
    setSelectedId(item.node.node_id);
    const position = positions.get(item.node.node_id);
    if (!position) return;
    graphContainerRef.current?.scrollTo({
      left: Math.max(0, (position.x - 80) * zoom),
      top: Math.max(0, (position.y - 80) * zoom),
      behavior: "smooth",
    });
  };
  const downloadPdf = async () => {
    if (!runId || pdfLoading) return;
    setPdfLoading(true);
    setPdfError(null);
    try {
      const response = await fetch(`/api/admin/answer-graphs/${runId}/pdf`, {
        cache: "no-store",
      });
      if (!response.ok)
        throw new Error(`PDF download failed (${response.status}).`);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = `answer-execution-${messageId}.pdf`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (failure) {
      setPdfError(
        failure instanceof Error ? failure.message : "Could not download PDF."
      );
    } finally {
      setPdfLoading(false);
    }
  };

  if (!user) return <p className="p-6">Loading account…</p>;
  if (!isAdmin) return <p className="p-6">Administrator access required.</p>;
  if (!graphUrl) return <p className="p-6">Invalid message ID.</p>;
  return (
    <main className="flex h-[calc(100vh-5rem)] min-h-[600px] flex-col p-5">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Answer execution</h1>
          <p className="text-sm opacity-70">Assistant message {messageId}</p>
          {run && (
            <p className="mt-1 text-sm">
              {run.status} · Capture {run.capture_status || "unknown"} ·{" "}
              {run.model_name || "model unknown"} · {nodes.length} operations
              {nodeOffset !== null ? "+" : ""}
              {run.capture_error ? ` · ${run.capture_error}` : ""}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {runId && (
            <Button
              size="sm"
              prominence="secondary"
              disabled={
                pdfLoading ||
                run?.status === "RUNNING" ||
                run?.status === "FINALIZING"
              }
              onClick={downloadPdf}
            >
              {pdfLoading ? "Preparing PDF…" : "Download PDF"}
            </Button>
          )}
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
      {pdfError && <p role="alert">{pdfError}</p>}
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
            className="min-w-0 flex-1 overflow-auto rounded-xl border border-border-01 bg-background-neutral-01"
            aria-label="Agent execution graph"
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
                {Array.from(new Set(layout.nodes.map((item) => item.rank))).map(
                  (rank) => (
                    <span
                      key={rank}
                      className="absolute top-1 text-xs font-semibold uppercase tracking-wide opacity-65"
                      style={{ left: 40 + rank * (GRAPH_NODE_WIDTH + 96) }}
                    >
                      Stage {rank + 1}
                    </span>
                  )
                )}
                {layout.lanes.map((lane) => (
                  <div
                    key={lane.agent}
                    className="absolute left-4 rounded-xl border border-border-02 bg-background-neutral-02/50"
                    style={{
                      top: lane.y,
                      width: layout.width - 32,
                      height: lane.height,
                    }}
                  >
                    <span className="absolute left-4 top-3 text-xs font-semibold uppercase tracking-wide opacity-70">
                      {lane.agent}
                    </span>
                  </div>
                ))}
                <svg
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0"
                  width={layout.width}
                  height={layout.height}
                >
                  <defs>
                    <marker
                      id="graph-arrow"
                      markerWidth="8"
                      markerHeight="8"
                      refX="7"
                      refY="4"
                      orient="auto"
                    >
                      <path d="M 0 0 L 8 4 L 0 8 z" fill="#94a3b8" />
                    </marker>
                    <marker
                      id="graph-data-arrow"
                      markerWidth="8"
                      markerHeight="8"
                      refX="7"
                      refY="4"
                      orient="auto"
                    >
                      <path d="M 0 0 L 8 4 L 0 8 z" fill="#3b82f6" />
                    </marker>
                  </defs>
                  {visibleEdges.map((edge, index) => {
                    const source = positions.get(edge.from_node_id)!;
                    const target = positions.get(edge.to_node_id)!;
                    const x1 = source.x + GRAPH_NODE_WIDTH;
                    const x2 = target.x;
                    const y1 = source.y + GRAPH_NODE_HEIGHT / 2;
                    const y2 = target.y + GRAPH_NODE_HEIGHT / 2;
                    return (
                      <path
                        key={`${edge.kind}:${edge.from_node_id}:${edge.to_node_id}:${index}`}
                        d={`M ${x1} ${y1} C ${x1 + 46} ${y1}, ${x2 - 46} ${y2}, ${x2 - 8} ${y2}`}
                        fill="none"
                        stroke={edge.kind === "data" ? "#3b82f6" : "#94a3b8"}
                        strokeWidth={edge.kind === "data" ? 2.5 : 1.5}
                        strokeDasharray={
                          edge.kind === "order"
                            ? "3 5"
                            : edge.kind === "parent"
                              ? "6 4"
                              : undefined
                        }
                        markerEnd={
                          edge.kind === "order"
                            ? undefined
                            : edge.kind === "data"
                              ? "url(#graph-data-arrow)"
                              : "url(#graph-arrow)"
                        }
                      />
                    );
                  })}
                </svg>
                {renderedNodes.map((item) => {
                  const node = item.node;
                  return (
                    <article
                      key={node.node_id}
                      className={`absolute flex flex-col rounded-xl border-l-4 p-3 shadow-sm ring-1 ring-border-01 ${PHASE_STYLES[item.phase]} ${selectedId === node.node_id ? "ring-2 ring-blue-500" : ""}`}
                      style={{
                        left: item.x,
                        top: item.y,
                        width: GRAPH_NODE_WIDTH,
                        height: GRAPH_NODE_HEIGHT,
                      }}
                    >
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span className="rounded bg-background-neutral-00 px-1.5 py-0.5 font-bold">
                          #{item.sequence}
                        </span>
                        <span className="font-semibold uppercase opacity-70">
                          {item.phase}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="mt-2 truncate text-left text-sm font-semibold hover:underline"
                        title={node.operation}
                        onClick={() => setSelectedId(node.node_id)}
                        aria-label={`Step ${item.sequence}: ${node.operation}`}
                        aria-pressed={selectedId === node.node_id}
                      >
                        {graphOperationLabel(node)}
                      </button>
                      <p
                        className="mt-1 truncate text-xs opacity-70"
                        title={item.agent}
                      >
                        {item.agent}
                      </p>
                      <p className="mt-auto truncate text-xs opacity-70">
                        {node.status} ·{" "}
                        {item.durationMs === null
                          ? "running"
                          : `${item.durationMs} ms`}
                        {item.callNumber > 1
                          ? ` · call ${item.callNumber}`
                          : ""}
                      </p>
                    </article>
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
                {searchResults.map((item) => (
                  <li key={item.node.node_id}>
                    <Button
                      size="sm"
                      prominence="tertiary"
                      width="full"
                      onClick={() => jumpToSequence(item.sequence)}
                    >
                      {`#${item.sequence} ${item.node.operation} · ${item.agent}`}
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
                <p className="mt-2 break-all text-sm font-semibold">
                  {selected ? `#${selected.sequence} · ` : ""}
                  {detail.node.operation}
                </p>
                <p className="text-xs opacity-70">
                  {detail.node.started_at} · {detail.node.status}
                </p>
                {selected && (
                  <div className="mt-3 rounded-lg border border-border-01 p-3 text-xs">
                    <p>Agent: {selected.agent}</p>
                    <p className="mt-1">Phase: {selected.phase}</p>
                    {selected.parentSequence !== null && (
                      <button
                        type="button"
                        className="mt-2 text-blue-500 hover:underline"
                        onClick={() => jumpToSequence(selected.parentSequence!)}
                      >
                        Parent: step #{selected.parentSequence}
                      </button>
                    )}
                    {selected.dataInputSequences.length > 0 && (
                      <p className="mt-1">
                        Data from:{" "}
                        {selected.dataInputSequences
                          .map((sequence) => `#${sequence}`)
                          .join(", ")}
                      </p>
                    )}
                  </div>
                )}
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
            {nodes.length} ordered operations · {edges.length} recorded data
            links
          </span>
          {nodes.length > 0 && (
            <span className="ml-auto text-xs opacity-70">
              Dashed: parent · Blue: recorded data · Dotted: same-agent order
            </span>
          )}
        </footer>
      )}
    </main>
  );
}
