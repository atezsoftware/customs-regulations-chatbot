const REQUEST_TIMEOUT_MS = 5_000;

export async function requestSupersearchStop(sessionId: string): Promise<void> {
  const response = await fetch(
    `/api/chat/stop-chat-session/${encodeURIComponent(sessionId)}`,
    { method: "POST", signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) }
  );
  if (!response.ok) {
    throw new Error(`Durdurma isteği gönderilemedi (${response.status}).`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export async function supersearchRunFinished(
  sessionId: string,
  assistantMessageId: number
): Promise<boolean> {
  const response = await fetch(
    `/api/chat/get-chat-session/${encodeURIComponent(sessionId)}`,
    { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) }
  );
  if (!response.ok) {
    throw new Error(`Çalışma durumu alınamadı (${response.status}).`);
  }
  const session: unknown = await response.json();
  if (
    !isRecord(session) ||
    session.chat_session_id !== sessionId ||
    session.current_run !== null ||
    !Array.isArray(session.messages)
  ) {
    return false;
  }
  const assistants = session.messages.filter(
    (message: unknown) =>
      isRecord(message) && message.message_type === "assistant"
  );
  const index = assistants.findIndex(
    (message: unknown) =>
      isRecord(message) && message.message_id === assistantMessageId
  );
  const message: unknown = assistants[index];
  if (!isRecord(message)) return false;
  if (typeof message.error === "string" && message.error.length > 0)
    return true;
  if (!Array.isArray(session.packets)) return false;
  const packets: unknown = session.packets[index];
  return (
    Array.isArray(packets) &&
    packets.some(
      (packet: unknown) =>
        isRecord(packet) &&
        isRecord(packet.obj) &&
        packet.obj.type === "asv3_progress" &&
        packet.obj.workflow === "supersearch" &&
        packet.obj.task_id == null &&
        ["completed", "failed", "cancelled"].includes(String(packet.obj.status))
    )
  );
}
