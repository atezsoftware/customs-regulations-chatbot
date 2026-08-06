import { contextualSetupStatusAfterSave } from "@/lib/indexing/contextual";

describe("contextualSetupStatusAfterSave", () => {
  it("keeps the first-index gate active while adopting the explicitly saved model", () => {
    expect(
      contextualSetupStatusAfterSave(
        {
          required: true,
          enabled: false,
          model_configuration_id: null,
        },
        42
      )
    ).toEqual({
      required: true,
      enabled: true,
      model_configuration_id: 42,
    });
  });
});
