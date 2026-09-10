import { readFileSync } from "node:fs";
import { test } from "@playwright/test";
import { ChatPage } from "../chat/ChatPage";
import { AnnexUpdatesPage } from "../pages/AnnexUpdatesPage";

test.describe.configure({ retries: 0 });

interface PreparedAnnexFixture {
  documentSetName: string;
  batchId: number;
  reviewSha256: string;
  screenshotPath: string;
}

test("approves the actual prepared linked annex once @annex", async ({
  page,
}) => {
  const fixturePath = process.env.ANNEX_ACCEPTANCE_EVIDENCE;
  test.skip(
    !fixturePath,
    "Requires the owned worker-prepared linked annex fixture."
  );
  const fixture: PreparedAnnexFixture = JSON.parse(
    readFileSync(fixturePath!, "utf8")
  );
  const updates = new AnnexUpdatesPage(page);
  await updates.openBatch(fixture.documentSetName, fixture.batchId);
  await updates.expectFrozenReview(fixture.reviewSha256);
  await updates.screenshot(fixture.screenshotPath);
  await updates.approveOnce();
  await updates.screenshot(
    fixture.screenshotPath.replace(/\.png$/, "-approved.png")
  );
});

test("opens both actual dated annex chat citations @annex", async ({
  page,
}) => {
  const fixturePath = process.env.ANNEX_ACCEPTANCE_CHAT_EVIDENCE;
  test.skip(!fixturePath, "Requires the owned actual dated chat evidence.");
  const fixture: { chat_ids: string[] } = JSON.parse(
    readFileSync(fixturePath!, "utf8")
  );
  const chat = new ChatPage(page);
  for (const [index, rate] of ["5%", "7%"].entries()) {
    await chat.openSavedCitation(
      fixture.chat_ids[index]!,
      rate,
      "Temsili Oran Yonetmeligi.pdf",
      fixturePath!.replace(/\.json$/, `-citation-${index}.png`)
    );
  }
});
