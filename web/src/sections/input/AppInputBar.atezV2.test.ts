import { sendMessage } from "@/app/app/services/lib";

const originalFetch = global.fetch;
afterEach(() => {
  global.fetch = originalFetch;
});

it.each([
  [false, false, false, false, "deep"],
  [true, false, false, true, "normal"],
  [false, true, false, false, "deep"],
  [true, true, false, false, "deep"],
  [false, false, true, true, "experimental"],
  [true, false, true, true, "experimental"],
  [false, true, true, false, "deep"],
])(
  "sends mutually exclusive research modes (%s, %s, %s)",
  async (single, deep, experimental, expectedASv3, profile) => {
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
