import {
  ASv3Progress,
  Packet,
  PacketType,
} from "@/app/app/services/streamingModels";

export const MAX_ASV3_PROGRESS_PACKETS = 256;
const RECENT_PROGRESS_PACKETS = 64;

/** Compact presentation history only; every source, citation and answer packet survives. */
export function compactASv3ProgressPackets(packets: Packet[]): Packet[] {
  const progressIndexes: number[] = [];
  const latestByScope = new Map<string, number>();
  let runId: string | undefined;
  for (let index = 0; index < packets.length; index++) {
    const packet = packets[index];
    if (packet?.obj.type !== PacketType.ASV3_PROGRESS) continue;
    const event = packet.obj as ASv3Progress;
    if (runId === undefined) runId = event.run_id;
    progressIndexes.push(index);
    if (event.run_id !== runId) continue;
    const scope = event.task_id ? `task:${event.task_id}` : "coordinator";
    const previousIndex = latestByScope.get(scope);
    const previous =
      previousIndex === undefined
        ? undefined
        : (packets[previousIndex]?.obj as ASv3Progress);
    if (!previous || event.sequence > previous.sequence)
      latestByScope.set(scope, index);
  }
  if (progressIndexes.length <= MAX_ASV3_PROGRESS_PACKETS) return packets;
  const retained = new Set([
    progressIndexes[0]!,
    ...progressIndexes.slice(-RECENT_PROGRESS_PACKETS),
    ...Array.from(latestByScope.values()),
  ]);
  return packets.filter(
    (packet, index) =>
      packet.obj.type !== PacketType.ASV3_PROGRESS || retained.has(index)
  );
}
