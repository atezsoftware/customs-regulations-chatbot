import { buildUsageExportUrl } from "./userUsage";

describe("buildUsageExportUrl", () => {
  it("uses the selected calendar dates for the per-user export", () => {
    const from = new Date(2026, 5, 1);
    const to = new Date(2026, 5, 14);
    const expectedPeriodTo = new Date(2026, 5, 14, 23, 59, 59, 999);

    expect(buildUsageExportUrl({ from, to })).toBe(
      `/api/admin/usage/export?period_from=${encodeURIComponent(
        from.toISOString()
      )}&period_to=${encodeURIComponent(expectedPeriodTo.toISOString())}`
    );
  });
});
