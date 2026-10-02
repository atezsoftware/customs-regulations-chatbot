import {
  ASv3Progress,
  Packet,
  PacketType,
} from "@/app/app/services/streamingModels";

export interface ASv3ProgressState {
  runId: string | null;
  header: ASv3Progress | null;
  history: ASv3Progress[];
  tasks: Map<string, ASv3Progress>;
  seenEvents: Set<string>;
  latestSequences: Map<string, number>;
  terminal: boolean;
}

const TERMINAL_STATUSES = new Set<ASv3Progress["status"]>([
  "completed",
  "failed",
  "cancelled",
]);

export const MAX_ASV3_COORDINATOR_STEPS = 32;

export function createASv3ProgressState(): ASv3ProgressState {
  return {
    runId: null,
    header: null,
    history: [],
    tasks: new Map(),
    seenEvents: new Set(),
    latestSequences: new Map(),
    terminal: false,
  };
}

/** Each task advances independently; a sibling completing cannot close it. */
export function applyASv3Progress(
  state: ASv3ProgressState,
  event: ASv3Progress
): ASv3ProgressState {
  if (
    state.terminal ||
    (state.runId !== null && state.runId !== event.run_id) ||
    state.seenEvents.has(event.event_id)
  ) {
    return state;
  }
  const scope = event.task_id ? `task:${event.task_id}` : "coordinator";
  const latestSequence = state.latestSequences.get(scope) ?? -1;
  if (event.sequence <= latestSequence) return state;

  const previousTask = event.task_id
    ? state.tasks.get(event.task_id)
    : undefined;
  if (previousTask && TERMINAL_STATUSES.has(previousTask.status)) return state;

  const tasks = new Map(state.tasks);
  if (event.task_id) tasks.set(event.task_id, event);
  const terminal = !event.task_id && TERMINAL_STATUSES.has(event.status);
  let history = state.history;
  if (!event.task_id) {
    const previous = history[history.length - 1];
    const samePublicUpdate =
      previous?.title === event.title &&
      (previous.message ?? "") === (event.message ?? "") &&
      previous.status === event.status &&
      previous.language === event.language;
    history = samePublicUpdate
      ? [...history.slice(0, -1), event]
      : [...history.slice(-(MAX_ASV3_COORDINATOR_STEPS - 1)), event];
  }
  if (terminal) {
    // Closing the run revokes work that has not produced a terminal task event.
    for (const [id, task] of Array.from(tasks)) {
      if (!TERMINAL_STATUSES.has(task.status)) {
        tasks.set(id, { ...task, status: "cancelled" });
      }
    }
  }
  return {
    runId: state.runId ?? event.run_id,
    header: !event.task_id || !state.header ? event : state.header,
    history,
    tasks,
    seenEvents: new Set([
      ...Array.from(state.seenEvents).slice(-255),
      event.event_id,
    ]),
    latestSequences: new Map(state.latestSequences).set(scope, event.sequence),
    terminal,
  };
}

export function collectASv3Progress(packets: Packet[]): ASv3ProgressState {
  return packets.reduce((state, packet) => {
    if (packet.obj.type !== PacketType.ASV3_PROGRESS) return state;
    return applyASv3Progress(state, packet.obj as ASv3Progress);
  }, createASv3ProgressState());
}
