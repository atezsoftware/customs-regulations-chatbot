/** A listed or merely enabled web tool never grants permission to leave the corpus. */
export function explicitASv3ExternalConsent(
  asv3: boolean,
  selectedToolIds: readonly number[],
  tools: readonly { id: number; in_code_tool_id: string | null }[],
  disabledToolIds: readonly number[]
): boolean {
  return (
    asv3 &&
    tools.some(
      (tool) =>
        tool.in_code_tool_id === "WebSearchTool" &&
        selectedToolIds.includes(tool.id) &&
        !disabledToolIds.includes(tool.id)
    )
  );
}
