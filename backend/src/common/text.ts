/**
 * Postgres text columns reject embedded NUL bytes outright
 * (invalid byte sequence for encoding "UTF8": 0x00), which a stray null
 * byte in a pasted message or LLM output would otherwise turn into an
 * unhandled 500 on message creation/streaming. Strip them before any text
 * makes it into a content/text-type column.
 */
const NUL = String.fromCharCode(0);

export function stripNulBytes(value: string): string {
  return value.split(NUL).join('');
}
