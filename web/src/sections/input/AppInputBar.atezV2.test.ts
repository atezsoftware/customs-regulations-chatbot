import { sendMessage } from "@/app/app/services/lib";

const originalFetch = global.fetch;
afterEach(() => {
  global.fetch = originalFetch;
});

it.each([
  [false, false, false, "deep"],
  [true, false, true, "normal"],
  [false, true, false, "deep"],
  [true, true, false, "deep"],
])(
  "sends mutually exclusive research modes (%s, %s)",
  async (single, deep, expectedASv3, profile) => {
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
    expect(payload.atez_search).toBe(false);
    expect(payload.atez_search_v2).toBe(false);
  }
);
