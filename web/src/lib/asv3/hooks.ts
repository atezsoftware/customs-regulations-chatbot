"use client";

import { useRef } from "react";
import {
  Packet,
  PacketType,
  ASv3Progress,
} from "@/app/app/services/streamingModels";
import {
  applyASv3Progress,
  createASv3ProgressState,
} from "@/lib/asv3/progress";

/** Consume appended packets once, including buffers updated in place. */
export function useASv3Progress(packets: Packet[], nodeId: number) {
  const cursor = useRef({
    nodeId,
    next: 0,
    packets,
    state: createASv3ProgressState(),
  });
  if (
    cursor.current.nodeId !== nodeId ||
    cursor.current.packets !== packets ||
    packets.length < cursor.current.next
  ) {
    cursor.current = {
      nodeId,
      next: 0,
      packets,
      state: createASv3ProgressState(),
    };
  }
  for (let index = cursor.current.next; index < packets.length; index++) {
    const packet = packets[index];
    if (packet?.obj.type === PacketType.ASV3_PROGRESS) {
      cursor.current.state = applyASv3Progress(
        cursor.current.state,
        packet.obj as ASv3Progress
      );
    }
  }
  cursor.current.next = packets.length;
  return cursor.current.state;
}
