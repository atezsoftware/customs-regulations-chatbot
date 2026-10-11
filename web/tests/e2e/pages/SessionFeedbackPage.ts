import {
  expect,
  type Locator,
  type Page,
  type Response,
} from "@playwright/test";
import type { SessionFeedbackRating } from "@/app/app/interfaces";

const RATING_LABELS: Record<SessionFeedbackRating, string> = {
  1: "Yanlış",
  2: "Eksik",
  3: "Kısmi doğru",
  4: "Doğru",
};

export class SessionFeedbackPage {
  readonly page: Page;
  readonly openButton: Locator;
  readonly dialog: Locator;
  readonly explanation: Locator;
  readonly saveButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.openButton = page.getByRole("button", {
      name: "Sohbeti değerlendir",
      exact: true,
    });
    this.dialog = page.getByRole("dialog", {
      name: /^Sohbeti değerlendir/,
    });
    this.explanation = this.dialog.getByLabel("Açıklama (opsiyonel)", {
      exact: true,
    });
    this.saveButton = this.dialog.getByRole("button", {
      name: "Kaydet",
      exact: true,
    });
  }

  async goto(sessionId: string): Promise<void> {
    await this.page.goto(`/app?chatId=${encodeURIComponent(sessionId)}`);
    await expect(this.openButton).toBeEnabled();
  }

  async reload(): Promise<void> {
    await this.page.reload();
    await expect(this.openButton).toBeEnabled();
  }

  async open(): Promise<void> {
    await this.openButton.click();
    await expect(this.dialog).toBeVisible();
  }

  rating(rating: SessionFeedbackRating): Locator {
    return this.dialog.getByRole("radio", {
      name: `${rating} yıldız: ${RATING_LABELS[rating]}`,
      exact: true,
    });
  }

  async expectOptions(): Promise<void> {
    await expect(
      this.dialog
        .getByRole("radiogroup", { name: "Sohbet puanı" })
        .getByRole("radio")
    ).toHaveCount(4);
    for (const rating of [1, 2, 3, 4] as const) {
      await expect(this.rating(rating)).toBeVisible();
      await expect(
        this.dialog.getByText(RATING_LABELS[rating], { exact: true })
      ).toBeVisible();
    }
  }

  async choose(rating: SessionFeedbackRating): Promise<void> {
    await this.rating(rating).click();
    await expect(this.rating(rating)).toBeChecked();
  }

  async moveSelection(
    from: SessionFeedbackRating,
    to: SessionFeedbackRating
  ): Promise<void> {
    await this.rating(from).press("ArrowRight");
    await expect(this.rating(to)).toBeChecked();
    await expect(this.rating(to)).toBeFocused();
  }

  async fillExplanation(text: string): Promise<void> {
    await this.explanation.fill(text);
  }

  async expectDraft(
    rating: SessionFeedbackRating,
    text: string
  ): Promise<void> {
    await expect(this.rating(rating)).toBeChecked();
    await expect(this.explanation).toHaveValue(text);
  }

  async expectSaveDisabled(): Promise<void> {
    await expect(this.saveButton).toBeDisabled();
  }

  async expectHeaderRating(rating: SessionFeedbackRating): Promise<void> {
    await expect(this.openButton).toHaveText(
      `${rating}/4 ${RATING_LABELS[rating]}`
    );
  }

  async submit(sessionId: string): Promise<Response> {
    const response = this.page.waitForResponse(
      (candidate) =>
        new URL(candidate.url()).pathname ===
          `/api/chat/chat-session/${sessionId}/feedback` &&
        candidate.request().method() === "PUT"
    );
    await this.saveButton.click();
    return response;
  }

  async save(sessionId: string): Promise<void> {
    expect((await this.submit(sessionId)).status()).toBe(200);
    await expect(this.dialog).toBeHidden();
  }

  async expectError(message: RegExp): Promise<void> {
    await expect(this.dialog.getByRole("alert")).toContainText(message);
    await expect(this.saveButton).toBeEnabled();
  }

  async expectStoredRating(rating: SessionFeedbackRating): Promise<void> {
    await expect(this.dialog).toContainText(
      `Kayıtlı puan: ${rating}/4 — ${RATING_LABELS[rating]}`
    );
  }

  async capture(path: string): Promise<void> {
    await this.dialog.screenshot({ path });
  }
}
