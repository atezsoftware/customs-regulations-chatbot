import { renderHook } from "@testing-library/react";
import { useASv3Progress } from "@/lib/asv3/hooks";
import type { Packet } from "@/app/app/services/streamingModels";

it("handles an appended stream buffer and resets when switching assistant messages", () => {
  const packets: Packet[] = [
    {
      placement: { turn_index: 0 },
      obj: {
        type: "asv3_progress",
        run_id: "run-1",
        event_id: "first",
        sequence: 1,
        language: "tr",
        phase: "research",
        status: "running",
        title: "İki ihtimali ayrı inceliyorum",
      },
    },
  ];
  const { result, rerender } = renderHook(
    ({ nodeId, version }) => {
      void version;
      return useASv3Progress(packets, nodeId);
    },
    { initialProps: { nodeId: 1, version: 0 } }
  );
  expect(result.current.header?.title).toBe("İki ihtimali ayrı inceliyorum");
  packets.push({
    placement: { turn_index: 1 },
    obj: {
      type: "asv3_progress",
      run_id: "run-1",
      event_id: "second",
      sequence: 2,
      language: "tr",
      phase: "completed",
      status: "completed",
      title: "Araştırma tamamlandı",
    },
  });
  rerender({ nodeId: 1, version: 1 });
  expect(result.current.terminal).toBe(true);
  packets.length = 0;
  rerender({ nodeId: 2, version: 2 });
  expect(result.current.runId).toBeNull();
});
