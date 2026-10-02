/**
 * Page Object Model for the main chat page (/app).
 *
 * Encapsulates locators and interactions shared across chat specs so that
 * individual tests remain declarative.
 */

import { type Page, type Locator, expect } from "@playwright/test";
import { expectElementScreenshot } from "@tests/e2e/utils/visualRegression";
import { InputBar } from "@tests/e2e/chat/InputBar";

export class ChatPage {
  readonly page: Page;
  readonly inputBar: InputBar;

  // Layout containers
  readonly container: Locator;
  readonly scrollContainer: Locator;

  // Message collections
  readonly humanMessages: Locator;
  readonly aiMessages: Locator;
  readonly usageLimitBanner: Locator;

  constructor(page: Page) {
    this.page = page;
    this.inputBar = new InputBar(page);
    this.container = page.locator("[data-main-container]");
    this.scrollContainer = page.getByTestId("chat-scroll-container");
    this.humanMessages = page.locator("#onyx-human-message");
    this.aiMessages = page.getByTestId("onyx-ai-message");
    this.usageLimitBanner = page.getByText(/you've reached the usage budget/i);
  }

  humanMessage(index = 0): Locator {
    return this.humanMessages.nth(index);
  }

  aiMessage(index = 0): Locator {
    return this.aiMessages.nth(index);
  }

  async goto(): Promise<void> {
    await this.page.goto("/app");
    await this.page.waitForLoadState("networkidle");
    await this.inputBar.textbox.waitFor({ state: "visible", timeout: 15000 });
  }

  async openSavedCitation(
    chatId: string,
    answer: string,
    citation: {
      semantic_identifier: string;
      document_id: string;
      chunk_ind: number;
    },
    screenshotPath: string
  ): Promise<void> {
    await this.page.goto(`/app?chatId=${encodeURIComponent(chatId)}`);
    const message = this.aiMessages.filter({ hasText: answer }).last();
    await expect(message).toBeVisible({ timeout: 30000 });
    const source = message
      .locator("p")
      .filter({ hasText: answer })
      .getByRole("button");
    await expect(source).toHaveCount(1);
    await expect(source).toBeInViewport({ timeout: 10000 });
    const content = this.page.waitForResponse((response) => {
      const url = new URL(response.url());
      return (
        url.pathname === "/api/document/chunk-info" &&
        url.searchParams.get("document_id") === citation.document_id &&
        url.searchParams.get("chunk_id") === String(citation.chunk_ind)
      );
    });
    await source.click({ timeout: 10000 });
    expect((await content).ok()).toBe(true);
    await expect(this.page.getByRole("dialog")).toContainText(answer);
    await this.page.screenshot({ path: screenshotPath, fullPage: true });
  }

  async expandASv3Progress(title: string): Promise<void> {
    const panel = this.page.getByRole("region", { name: "ASv3" }).last();
    const toggle = panel.getByRole("button", { name: title, exact: true });
    await expect(toggle).toHaveAttribute("aria-expanded", "false");
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
  }

  async expectASv3Pending(reducedMotion: boolean): Promise<void> {
    const panel = this.page.getByRole("region", { name: "ASv3" }).last();
    await expect(panel).toHaveText("ASv3");
    await expect(
      this.page.getByText("Thinking...", { exact: true })
    ).toHaveCount(0);
    if (reducedMotion) {
      const title = panel.getByTestId("asv3-progress-title");
      await expect(title).toHaveCSS("animation-name", "none");
      await expect(title.getByText("ASv3", { exact: true })).not.toHaveCSS(
        "color",
        "rgba(0, 0, 0, 0)"
      );
    }
  }

  async expectASv3TerminalPresentation(): Promise<void> {
    const panel = this.page.getByRole("region", { name: "ASv3" }).last();
    await expect(panel).toHaveAttribute("aria-busy", "false");
    await expect(panel.getByTestId("asv3-task-loading")).toHaveCount(0);
    await expect(panel.getByTestId("asv3-progress-title")).toHaveCSS(
      "animation-name",
      "none"
    );
  }

  async captureASv3Progress(path: string): Promise<void> {
    const panel = this.page.getByRole("region", { name: "ASv3" }).last();
    await expect(panel).toBeVisible();
    await panel.screenshot({ path });
  }

  async expectASv3Progress(
    language: string,
    title: string,
    tasks: string[]
  ): Promise<void> {
    const panel = this.page.getByRole("region", { name: "ASv3" }).last();
    await expect(panel).toHaveAttribute("lang", language);
    await expect(panel).toContainText(title);
    await expect(panel.getByTestId("asv3-task")).toHaveCount(tasks.length);
    for (const task of tasks) await expect(panel).toContainText(task);
    await expect(panel).not.toContainText("read_provision");
    await expect(panel).not.toContainText("spawn_researcher");
    await expect(panel).not.toContainText("Research Task");
  }

  async expectASv3TaskStatus(taskId: string, status: string): Promise<void> {
    await expect(
      this.page.locator(`[data-testid="asv3-task"][data-task-id="${taskId}"]`)
    ).toHaveAttribute("data-status", status);
  }

  async expectASv3CitationTarget(
    answer: string,
    documentId: string,
    chunkInd: number,
    previewUrl?: string
  ): Promise<void> {
    const message = this.aiMessages.filter({ hasText: answer }).last();
    const source = message
      .locator("p")
      .filter({ hasText: answer })
      .getByRole("button");
    await expect(source).toHaveCount(1);
    const response = this.page.waitForResponse((response) => {
      const url = new URL(response.url());
      if (previewUrl) return url.pathname === previewUrl;
      return (
        url.pathname === "/api/document/chunk-info" &&
        url.searchParams.get("document_id") === documentId &&
        url.searchParams.get("chunk_id") === String(chunkInd)
      );
    });
    await source.click();
    expect((await response).ok()).toBe(true);
    await expect(this.page.getByRole("dialog")).toContainText(answer);
  }

  async scrollTo(position: "top" | "bottom"): Promise<void> {
    await this.scrollContainer.evaluate(async (el, pos) => {
      el.scrollTo({ top: pos === "top" ? 0 : el.scrollHeight });
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
    }, position);
  }

  async screenshotContainer(name: string): Promise<void> {
    await expect(this.container).toBeVisible();
    if ((await this.scrollContainer.count()) > 0) {
      await this.scrollTo("bottom");
    }
    await expectElementScreenshot(this.container, { name });
  }

  /**
   * Captures two screenshots of the chat container for long-content tests:
   * one scrolled to the top and one scrolled to the bottom. Ensures
   * consistent scroll positions regardless of whether the page was just
   * navigated to (top) or just finished streaming (bottom).
   */
  async screenshotContainerTopAndBottom(name: string): Promise<void> {
    await expect(this.container).toBeVisible();

    await this.scrollTo("top");
    await expectElementScreenshot(this.container, { name: `${name}-top` });

    await this.scrollTo("bottom");
    await expectElementScreenshot(this.container, { name: `${name}-bottom` });
  }

  // ---------------------------------------------------------------------------
  // Message assertions
  // ---------------------------------------------------------------------------

  async expectHumanMessage(text: string, index = 0): Promise<void> {
    await expect(this.humanMessage(index)).toContainText(text);
  }

  async expectNoHumanMessages(): Promise<void> {
    await expect(this.humanMessages).toHaveCount(0);
  }

  async sendUntilUsageLimit(maxTurns: number): Promise<void> {
    for (
      let turn = 0;
      turn < maxTurns && !(await this.usageLimitBanner.isVisible());
      turn++
    ) {
      await this.inputBar.fill(`write a few sentences about topic ${turn}`);
      await this.inputBar.send();
      await Promise.race([
        this.usageLimitBanner
          .waitFor({ state: "visible", timeout: 45_000 })
          .catch(() => {}),
        this.aiMessage(turn)
          .waitFor({ state: "visible", timeout: 45_000 })
          .catch(() => {}),
      ]);
    }
  }

  async expectAccountUsageLimit(): Promise<void> {
    await expect(this.usageLimitBanner).toBeVisible();
    await expect(this.page.getByText(/your account/i)).toBeVisible();
  }
}
