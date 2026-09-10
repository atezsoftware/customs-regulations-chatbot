import { expect, type Locator, type Page } from "@playwright/test";

export class AnnexUpdatesPage {
  readonly page: Page;
  readonly review: Locator;

  constructor(page: Page) {
    this.page = page;
    this.review = page.getByRole("article").filter({
      has: page.getByRole("heading", { name: "Annex EK-1", exact: true }),
    });
  }

  async openBatch(documentSetName: string, batchId: number): Promise<void> {
    await this.page.goto("/admin/amendments", { timeout: 60000 });
    await this.page.getByRole("combobox").click();
    await this.page
      .getByRole("option", { name: documentSetName, exact: true })
      .click();
    await this.page
      .getByRole("button", { name: new RegExp(`^Batch #${batchId} `) })
      .click();
    await expect(this.review).toBeVisible();
  }

  async expectFrozenReview(hash: string): Promise<void> {
    await this.review.getByText("Audit details", { exact: true }).click();
    await expect(
      this.review.getByText(`Frozen review hash: ${hash}`, { exact: true })
    ).toBeVisible();
    await expect(
      this.review.getByRole("heading", { name: "Frozen original evidence" })
    ).toBeVisible();
    await expect(
      this.review.getByRole("link", { name: /^OLD original/ })
    ).toHaveCount(1);
    await expect(
      this.review.getByRole("link", { name: /^NEW original/ })
    ).toHaveCount(2);
    await expect(this.review.getByText(/^OLD: 5%/)).toBeVisible();
    await expect(this.review.getByText(/^NEW: 7%/)).toBeVisible();
    await expect(
      this.review.getByRole("heading", { name: "Final publication scope" })
    ).toBeVisible();
    await expect(
      this.review.getByText("Context reevaluation candidates", { exact: true })
    ).toBeVisible();
    await expect(
      this.review.getByText("Final embedding changes", { exact: true })
    ).toBeVisible();
  }

  async approveOnce(): Promise<void> {
    await this.review
      .getByRole("button", { name: "Approve group", exact: true })
      .click();
    await expect(
      this.review.getByText("Approved", { exact: true })
    ).toBeVisible({ timeout: 120000 });
    await expect(
      this.review.getByRole("button", { name: "Approve group", exact: true })
    ).toHaveCount(0);
  }

  async screenshot(path: string): Promise<void> {
    await this.page.screenshot({ path, fullPage: true });
  }
}
