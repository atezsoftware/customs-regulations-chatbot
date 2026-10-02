import { renderHook } from "@testing-library/react";
import type { Packet } from "@/app/app/services/streamingModels";
import { collectASv3Progress } from "@/lib/asv3/progress";
import { useASv3Progress } from "@/lib/asv3/hooks";
import { usePacketProcessor } from "@/app/app/message/messageComponents/timeline/hooks/usePacketProcessor";
import {
  createCitationPacket,
  createMessageStartPacket,
  createSearchToolDocumentsPacket,
  createSearchToolStartPacket,
} from "@/app/app/message/messageComponents/timeline/hooks/__tests__/testHelpers";
import {
  compactASv3ProgressPackets,
  MAX_ASV3_PROGRESS_PACKETS,
} from "./retention";

function progress(sequence: number, taskId?: string): Packet {
  return {
    placement: { turn_index: 0 },
    obj: {
      type: "asv3_progress",
      run_id: "run-1",
      event_id: `event-${sequence}`,
      sequence,
      language: "tr",
      phase: "research",
      status: "running",
      title: `Kaynak incelemesi ${sequence}`,
      task_id: taskId,
    },
  };
}

it("bounds long-running public history while preserving each latest task and all evidence", () => {
  let packets: Packet[] = [progress(0), progress(1, "idle-child")];
  const documents = createSearchToolDocumentsPacket(
    [{ document_id: "law", chunk_ind: 7 }],
    { turn_index: 1 }
  );
  const citation = createCitationPacket(1, "law", { turn_index: 2 }, 7);
  const message = createMessageStartPacket({ turn_index: 3 });
  packets.push(
    createSearchToolStartPacket({ turn_index: 1 }),
    documents,
    citation,
    message
  );
  for (let sequence = 2; sequence < 5000; sequence++) {
    packets.push(progress(sequence, sequence % 2 ? "active-child" : undefined));
    packets = compactASv3ProgressPackets(packets);
  }
  const progressPackets = packets.filter(
    (packet) => packet.obj.type === "asv3_progress"
  );
  expect(progressPackets.length).toBeLessThanOrEqual(MAX_ASV3_PROGRESS_PACKETS);
  expect(packets).toContain(documents);
  expect(packets).toContain(citation);
  expect(packets).toContain(message);
  expect(packets.indexOf(documents)).toBeLessThan(packets.indexOf(citation));
  const state = collectASv3Progress(packets);
  expect(state.tasks.get("idle-child")?.sequence).toBe(1);
  expect(state.tasks.get("active-child")?.sequence).toBe(4999);
  expect(state.header?.sequence).toBe(4998);
  expect(state.seenEvents.size).toBeLessThanOrEqual(256);
});

it("reprocesses replaced buffers without losing progress, exact citations, or native routes", () => {
  const nativeCitation = createCitationPacket(
    2,
    "original",
    { turn_index: 2 },
    -2
  );
  if (nativeCitation.obj.type === "citation_info")
    nativeCitation.obj.preview_url = "/api/asv3/citation/101/2";
  const packets: Packet[] = [
    progress(0),
    createSearchToolStartPacket({ turn_index: 1 }),
    createSearchToolDocumentsPacket(
      [{ document_id: "original", chunk_ind: -2 }],
      { turn_index: 1 }
    ),
    nativeCitation,
  ];
  const { result, rerender } = renderHook(
    ({ buffer }) => ({
      progress: useASv3Progress(buffer, 1),
      processor: usePacketProcessor(buffer, 1, true),
    }),
    { initialProps: { buffer: packets } }
  );
  const replacement = [...packets, progress(1, "child"), progress(2)];
  rerender({ buffer: replacement });
  expect(result.current.progress.tasks.get("child")?.title).toBe(
    "Kaynak incelemesi 1"
  );
  expect(result.current.processor.citations).toHaveLength(1);
  expect(result.current.processor.citations[0]?.preview_url).toBe(
    "/api/asv3/citation/101/2"
  );
  expect(result.current.processor.citationChunkMap).toEqual({ 2: -2 });
});
