import type { RegulatoryChunk } from "@/lib/regulatory/interfaces";

export function chunkEditRestriction(chunk: RegulatoryChunk): string | null {
  if (
    chunk.chunk_type === "hierarchical_aggregate" ||
    chunk.chunk_metadata.chunk_variant === "hierarchical_aggregate"
  ) {
    return "Bu chunk alt chunklardan türetilir. Değişiklik için kaynak alt chunkı düzenleyin.";
  }
  return null;
}
