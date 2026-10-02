import { explicitASv3ExternalConsent } from "./consent";
const tools = [
  { id: 7, in_code_tool_id: "WebSearchTool" },
  { id: 8, in_code_tool_id: "SearchTool" },
];
it("keeps listed web tools corpus-only until explicitly selected", () => {
  expect(explicitASv3ExternalConsent(true, [], tools, [])).toBe(false);
  expect(explicitASv3ExternalConsent(true, [8], tools, [])).toBe(false);
  expect(explicitASv3ExternalConsent(true, [7], tools, [])).toBe(true);
});
it("ignores stale, disabled and other workflow selections", () => {
  expect(explicitASv3ExternalConsent(false, [7], tools, [])).toBe(false);
  expect(explicitASv3ExternalConsent(true, [7], [], [])).toBe(false);
  expect(explicitASv3ExternalConsent(true, [7], tools, [7])).toBe(false);
});
