import { sendMessage } from "@/app/app/services/lib";

const originalFetch = global.fetch;
afterEach(() => {
  global.fetch = originalFetch;
});

it.each([
  [false, false, false, false, false, "deep", false],
  [true, false, false, false, true, "normal", false],
  [false, true, false, false, false, "deep", false],
  [true, true, false, false, false, "deep", false],
  [false, false, true, false, true, "experimental", false],
  [true, false, true, false, true, "experimental", false],
  [false, true, true, false, false, "deep", false],
  [false, false, false, true, true, "experimental", true],
  [true, false, false, true, true, "experimental", true],
  [false, false, true, true, true, "experimental", true],
  [false, true, false, true, false, "deep", false],
])(
  "sends mutually exclusive research modes (%s, %s, %s, %s)",
  async (
    single,
    deep,
    experimental,
    parallel,
    expectedASv3,
    profile,
    expectedParallel
  ) => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "captured" }),
    });
    await expect(
      sendMessage({
        message: "Soru",
        chatSessionId: "session-1",
        parentMessageId: null,
        filters: null,
        atezSearchV3: Boolean(single),
        deepResearch: Boolean(deep),
        experimentalResearch: Boolean(experimental),
        experimentalParallelResearch: Boolean(parallel),
        atezSearch: true,
        atezSearchV2: true,
      }).next()
    ).rejects.toThrow("captured");
    const payload = JSON.parse(
      String(jest.mocked(global.fetch).mock.calls[0]![1]?.body)
    );
    expect(payload.atez_search_v3).toBe(expectedASv3);
    expect(payload.deep_research).toBe(deep);
    expect(payload.asv3_research_profile).toBe(profile);
    if (expectedParallel) {
      expect(payload.asv3_parallel_research).toBe(true);
    } else {
      expect(payload).not.toHaveProperty("asv3_parallel_research");
    }
    expect(payload.atez_search).toBe(false);
    expect(payload.atez_search_v2).toBe(false);
  }
);
