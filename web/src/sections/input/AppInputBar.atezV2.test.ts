import { readFileSync } from "node:fs";
import { join } from "node:path";

test("Atez Search V2 keeps its label visible while unselected", () => {
  const source = readFileSync(join(__dirname, "AppInputBar.tsx"), "utf8");
  const start = source.indexOf("{showAtezSearch && toggleAtezSearchV2 && (");
  const end = source.indexOf("Atez Search V2", start);
  const v2Button = source.slice(start, end);

  expect(start).toBeGreaterThanOrEqual(0);
  expect(end).toBeGreaterThan(start);
  expect(v2Button).toContain("foldable={false}");
});
