import {
  applyASv3Progress,
  createASv3ProgressState,
  collectASv3Progress,
} from "@/lib/asv3/progress";
import type { ASv3Progress, Packet } from "@/app/app/services/streamingModels";

function event(
  sequence: number,
  changes: Partial<ASv3Progress> = {}
): ASv3Progress {
  return {
    type: "asv3_progress",
    run_id: "run-1",
    event_id: `event-${sequence}`,
    sequence,
    language: "tr",
    phase: "research",
    status: "running",
    title: "Kaynaklar inceleniyor",
    ...changes,
  };
}

it("keeps a parallel sibling active when a different task or placement finishes", () => {
  const packets: Packet[] = [
    { placement: { turn_index: 0 }, obj: event(1) },
    {
      placement: { turn_index: 1, tab_index: 0 },
      obj: event(2, { task_id: "a" }),
    },
    {
      placement: { turn_index: 1, tab_index: 1 },
      obj: event(3, { task_id: "b" }),
    },
    {
      placement: { turn_index: 2 },
      obj: event(4, { task_id: "a", status: "completed" }),
    },
    { placement: { turn_index: 3 }, obj: { type: "section_end" } },
  ];
  const state = collectASv3Progress(packets);
  expect(state.tasks.get("a")?.status).toBe("completed");
  expect(state.tasks.get("b")?.status).toBe("running");
  expect(state.terminal).toBe(false);
});

it("ignores duplicate, stale task, foreign run and post-terminal results", () => {
  let state = applyASv3Progress(createASv3ProgressState(), event(1));
  state = applyASv3Progress(state, event(3, { task_id: "a" }));
  expect(applyASv3Progress(state, event(3, { task_id: "a" }))).toBe(state);
  expect(
    applyASv3Progress(state, event(2, { task_id: "a", title: "Eski sonuç" }))
  ).toBe(state);
  expect(applyASv3Progress(state, event(4, { run_id: "other-run" }))).toBe(
    state
  );
  state = applyASv3Progress(
    state,
    event(5, { status: "cancelled", title: "Araştırma durduruldu" })
  );
  expect(state.tasks.get("a")?.status).toBe("cancelled");
  expect(state.terminal).toBe(true);
  expect(
    applyASv3Progress(state, event(6, { task_id: "a", status: "completed" }))
  ).toBe(state);
});

it("accepts interleaved lower sequence from a different task and preserves terminal tasks", () => {
  let state = applyASv3Progress(createASv3ProgressState(), event(1));
  state = applyASv3Progress(
    state,
    event(9, { task_id: "a", status: "completed" })
  );
  state = applyASv3Progress(state, event(8, { task_id: "b" }));
  expect(state.tasks.get("b")?.status).toBe("running");
  expect(applyASv3Progress(state, event(10, { task_id: "a" }))).toBe(state);
});
