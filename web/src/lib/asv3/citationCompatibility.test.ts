import {
  createInitialState,
  processPackets,
} from "@/app/app/message/messageComponents/timeline/hooks/packetProcessor";
import { collectASv3Progress } from "@/lib/asv3/progress";
import {
  createCitationPacket,
  createMessageStartPacket,
  createStopPacket,
  createSearchToolDocumentsPacket,
  createSearchToolStartPacket,
} from "@/app/app/message/messageComponents/timeline/hooks/__tests__/testHelpers";
import type { Packet } from "@/app/app/services/streamingModels";

it("keeps chunk-specific citations and final answer when progress interleaves, including replay", () => {
  const packets: Packet[] = [
    {
      placement: { turn_index: 0 },
      obj: {
        type: "asv3_progress",
        run_id: "run-1",
        event_id: "start",
        sequence: 1,
        language: "tr",
        phase: "research",
        status: "running",
        title: "Araştırma başladı",
      },
    },
    createSearchToolStartPacket({ turn_index: 1 }),
    createSearchToolDocumentsPacket(
      [
        { document_id: "law", chunk_ind: 7 },
        { document_id: "law", chunk_ind: 12 },
      ],
      { turn_index: 1 }
    ),
    createCitationPacket(1, "law", { turn_index: 2 }, 7),
    createCitationPacket(2, "law", { turn_index: 2 }, 12),
    createMessageStartPacket({ turn_index: 3 }),
    {
      placement: { turn_index: 4 },
      obj: {
        type: "asv3_progress",
        run_id: "run-1",
        event_id: "done",
        sequence: 2,
        language: "tr",
        phase: "completed",
        status: "completed",
        title: "Araştırma tamamlandı",
      },
    },
    createStopPacket(),
  ];
  for (const replay of [false, true]) {
    let state = createInitialState(1);
    if (!replay) state = processPackets(state, packets.slice(0, 4));
    state = processPackets(state, packets);
    expect(state.citationMap).toEqual({ 1: "law", 2: "law" });
    expect(state.citationChunkMap).toEqual({ 1: 7, 2: 12 });
    expect(state.citations).toHaveLength(2);
    expect(state.documentMap.size).toBe(2);
    expect(state.finalAnswerComing).toBe(true);
    expect(collectASv3Progress(packets).terminal).toBe(true);
  }
});

it("does not let operational progress occupy the final answer placement", () => {
  const packets: Packet[] = [
    {
      placement: { turn_index: 0 },
      obj: {
        type: "asv3_progress",
        run_id: "run-1",
        event_id: "start",
        sequence: 1,
        language: "tr",
        phase: "research",
        status: "running",
        title: "Kaynakları inceliyorum",
      },
    },
    createMessageStartPacket({ turn_index: 0 }),
    createCitationPacket(1, "law", { turn_index: 0 }, 7),
    {
      placement: { turn_index: 0 },
      obj: { type: "message_delta", content: "Yanıt [[1]]()" },
    },
    createStopPacket(),
  ];
  const state = processPackets(createInitialState(1), packets);
  expect(state.potentialDisplayGroups).toHaveLength(1);
  expect(
    state.potentialDisplayGroups[0]?.packets.some(
      (packet) => packet.obj.type === "message_delta"
    )
  ).toBe(true);
  expect(
    state.groupedPacketsMap
      .get("0-0")
      ?.some((packet) => packet.obj.type === "asv3_progress")
  ).toBe(false);
  expect(state.citationChunkMap).toEqual({ 1: 7 });
});
