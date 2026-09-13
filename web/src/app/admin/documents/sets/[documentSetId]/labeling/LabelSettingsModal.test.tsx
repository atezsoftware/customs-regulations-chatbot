import {
  act,
  render,
  screen,
  setupUser,
  waitFor,
} from "@tests/setup/test-utils";

import LabelSettingsModal from "@/app/admin/documents/sets/[documentSetId]/labeling/LabelSettingsModal";
import type { LabelSettingsSnapshot } from "@/lib/documentSetLabeling/interfaces";

const settingsUrl = "/api/manage/admin/document-set/7/labeling/label-settings";

const initialSnapshot: LabelSettingsSnapshot = {
  revision: 3,
  taxonomy_id: "taxonomy-v3",
  labels: [
    {
      id: "CUS.VALUE",
      name: "Customs valuation",
      description: "Rules used to determine customs value.",
    },
    {
      id: "TAX.VAT",
      name: "Import VAT",
      description: "VAT charged on imported goods.",
    },
  ],
  updated_at: "2026-09-13T10:00:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

afterEach(() => {
  jest.restoreAllMocks();
});

test("searches one editor and saves edits, additions, and deletions as one revision", async () => {
  const user = setupUser();
  const onClose = jest.fn();
  const onSaved = jest.fn();
  const savedSnapshot: LabelSettingsSnapshot = {
    ...initialSnapshot,
    revision: 4,
    taxonomy_id: "taxonomy-v4",
    labels: [
      {
        id: "CUS.VALUE",
        name: "Customs value",
        description: "Updated valuation definition.",
      },
      {
        id: "NEW.LABEL",
        name: "New label",
        description: "A new regulatory concept.",
      },
    ],
    updated_at: "2026-09-13T11:00:00Z",
  };
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockImplementation(async (input, init) => {
      expect(String(input)).toBe(settingsUrl);
      if (init?.method === "PUT") return jsonResponse(savedSnapshot);
      return jsonResponse(initialSnapshot);
    });

  render(
    <LabelSettingsModal documentSetId={7} onClose={onClose} onSaved={onSaved} />
  );

  expect(
    (await screen.findAllByText(/shared across all document sets/i)).length
  ).toBeGreaterThan(0);
  expect(screen.getByLabelText("Label ID")).toHaveAttribute("readonly");

  await user.clear(screen.getByLabelText("Label name"));
  await user.type(screen.getByLabelText("Label name"), "Customs value");
  await user.clear(screen.getByLabelText("Label description"));
  await user.type(
    screen.getByLabelText("Label description"),
    "Updated valuation definition."
  );

  await user.type(screen.getByLabelText("Search labels"), "VAT");
  await user.click(
    screen.getByRole("button", { name: /TAX\.VAT.*Import VAT/ })
  );
  await user.click(screen.getByRole("button", { name: "Delete label" }));

  await user.click(screen.getByRole("button", { name: "Add label" }));
  expect(screen.getByLabelText("Label ID")).not.toHaveAttribute("readonly");
  await user.type(screen.getByLabelText("Label ID"), "NEW.LABEL");
  await user.type(screen.getByLabelText("Label name"), "New label");
  await user.type(
    screen.getByLabelText("Label description"),
    "A new regulatory concept."
  );
  await user.click(screen.getByRole("button", { name: "Save changes" }));

  await waitFor(() => expect(onSaved).toHaveBeenCalledWith(savedSnapshot));
  expect(onClose).toHaveBeenCalledTimes(1);
  const saveCall = fetchSpy.mock.calls.find(
    ([, init]) => init?.method === "PUT"
  );
  expect(JSON.parse(String(saveCall?.[1]?.body))).toEqual({
    expected_revision: 3,
    labels: savedSnapshot.labels,
  });
});

test("preserves a conflicting draft until the user explicitly reloads", async () => {
  const user = setupUser();
  const latestSnapshot: LabelSettingsSnapshot = {
    ...initialSnapshot,
    revision: 4,
    labels: [
      {
        ...initialSnapshot.labels[0]!,
        description: "Definition saved by another administrator.",
      },
      initialSnapshot.labels[1]!,
    ],
  };
  let getRequests = 0;
  jest.spyOn(global, "fetch").mockImplementation(async (_input, init) => {
    if (init?.method === "PUT")
      return jsonResponse(
        { detail: "Label settings changed. Reload before saving." },
        409
      );
    getRequests += 1;
    return jsonResponse(getRequests === 1 ? initialSnapshot : latestSnapshot);
  });

  render(
    <LabelSettingsModal
      documentSetId={7}
      onClose={jest.fn()}
      onSaved={jest.fn()}
    />
  );
  const description = await screen.findByLabelText("Label description");
  await user.clear(description);
  await user.type(description, "My unsaved definition.");
  await user.click(screen.getByRole("button", { name: "Save changes" }));

  expect(
    await screen.findByText("Label settings changed. Reload before saving.")
  ).toBeInTheDocument();
  expect(screen.getByLabelText("Label description")).toHaveValue(
    "My unsaved definition."
  );

  await user.click(screen.getByRole("button", { name: "Reload labels" }));
  await waitFor(() =>
    expect(screen.getByLabelText("Label description")).toHaveValue(
      "Definition saved by another administrator."
    )
  );
});

test("locks edits, selection, and close actions while a save is in flight", async () => {
  const user = setupUser();
  const onClose = jest.fn();
  const onSaved = jest.fn();
  let resolveSave!: (response: Response) => void;
  const saveResponse = new Promise<Response>((resolve) => {
    resolveSave = resolve;
  });
  jest.spyOn(global, "fetch").mockImplementation(async (_input, init) => {
    if (init?.method === "PUT") return saveResponse;
    return jsonResponse(initialSnapshot);
  });

  render(
    <LabelSettingsModal documentSetId={7} onClose={onClose} onSaved={onSaved} />
  );
  const nameInput = await screen.findByLabelText("Label name");
  await user.clear(nameInput);
  await user.type(nameInput, "Updated while safe");
  await user.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await screen.findByText("Saving…")).toBeInTheDocument();
  expect(screen.getByLabelText("Search labels")).toBeDisabled();
  expect(screen.getByLabelText("Label name")).toBeDisabled();
  expect(screen.getByLabelText("Label description")).toBeDisabled();
  expect(screen.getByRole("button", { name: "Add label" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Delete label" })).toBeDisabled();
  expect(
    screen.getByRole("button", { name: /TAX\.VAT.*Import VAT/ })
  ).toBeDisabled();
  expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  expect(
    screen.queryByRole("button", { name: "Close" })
  ).not.toBeInTheDocument();
  expect(onClose).not.toHaveBeenCalled();

  await act(async () => {
    resolveSave(
      jsonResponse({
        ...initialSnapshot,
        revision: 4,
        labels: [
          { ...initialSnapshot.labels[0]!, name: "Updated while safe" },
          initialSnapshot.labels[1]!,
        ],
      })
    );
  });
  await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));
  expect(onClose).toHaveBeenCalledTimes(1);
});

test("keeps invalid new labels in the editor and does not call the API", async () => {
  const user = setupUser();
  const fetchSpy = jest
    .spyOn(global, "fetch")
    .mockResolvedValue(jsonResponse(initialSnapshot));

  render(
    <LabelSettingsModal
      documentSetId={7}
      onClose={jest.fn()}
      onSaved={jest.fn()}
    />
  );
  await screen.findByLabelText("Label description");
  await user.click(screen.getByRole("button", { name: "Add label" }));
  expect(screen.getByLabelText("Label ID")).toHaveAttribute("maxlength", "100");
  await user.type(screen.getByLabelText("Label ID"), "invalid id");
  await user.type(screen.getByLabelText("Label name"), "Invalid label");
  await user.type(screen.getByLabelText("Label description"), "Definition");
  await user.click(screen.getByRole("button", { name: "Save changes" }));

  expect(
    await screen.findByText(
      /letters, numbers, underscores, periods, colons, and hyphens/i
    )
  ).toBeInTheDocument();
  expect(screen.getByLabelText("Label ID")).toHaveValue("invalid id");
  expect(fetchSpy).toHaveBeenCalledTimes(1);
});
