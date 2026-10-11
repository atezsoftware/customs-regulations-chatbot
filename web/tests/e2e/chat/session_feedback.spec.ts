import { expect, test, type APIRequestContext } from "@playwright/test";
import { SessionFeedbackPage } from "@tests/e2e/pages/SessionFeedbackPage";
import type {
  BackendChatSession,
  SessionFeedbackResponse,
} from "@/app/app/interfaces";

async function createSession(request: APIRequestContext): Promise<string> {
  const response = await request.post("/api/chat/create-chat-session", {
    data: { persona_id: 0, description: "Session feedback E2E" },
  });
  expect(response.status()).toBe(200);
  const body = (await response.json()) as { chat_session_id: string };
  return body.chat_session_id;
}

async function readFeedback(
  request: APIRequestContext,
  sessionId: string
): Promise<SessionFeedbackResponse | null> {
  const response = await request.get(`/api/chat/get-chat-session/${sessionId}`);
  expect(response.status()).toBe(200);
  return ((await response.json()) as BackendChatSession).session_feedback;
}

test.describe("Session feedback", () => {
  let sessionId: string;

  test.beforeEach(async ({ page }) => {
    sessionId = await createSession(page.request);
  });

  test.afterEach(async ({ page }) => {
    await page.unrouteAll({ behavior: "ignoreErrors" });
    if (sessionId) {
      const response = await page.request.delete(
        `/api/chat/delete-chat-session/${sessionId}`
      );
      expect(response.ok()).toBe(true);
    }
  });

  test("four labeled stars support keyboard selection, optional text and persisted updates", async ({
    page,
  }, testInfo) => {
    const feedback = new SessionFeedbackPage(page);
    await feedback.goto(sessionId);
    await feedback.open();
    await feedback.expectOptions();
    await feedback.expectSaveDisabled();
    await feedback.choose(1);
    await feedback.moveSelection(1, 2);
    await feedback.moveSelection(2, 3);
    await feedback.moveSelection(3, 4);
    await feedback.save(sessionId);
    await feedback.expectHeaderRating(4);
    expect(await readFeedback(page.request, sessionId)).toMatchObject({
      rating: 4,
      label: "Doğru",
      feedback_text: null,
      version: 1,
    });

    await feedback.open();
    await feedback.expectDraft(4, "");
    await feedback.choose(3);
    const explanation = "İlgili istisna eksik. <script>plain text</script>";
    await feedback.fillExplanation(explanation);
    await feedback.save(sessionId);
    await feedback.reload();
    await feedback.expectHeaderRating(3);
    await feedback.open();
    await feedback.expectDraft(3, explanation);
    expect(await readFeedback(page.request, sessionId)).toMatchObject({
      rating: 3,
      label: "Kısmi doğru",
      feedback_text: explanation,
      version: 2,
    });
    await feedback.capture(testInfo.outputPath("session-feedback.png"));
  });

  test("failed save keeps the selected rating and explanation for a successful retry", async ({
    page,
  }) => {
    const feedback = new SessionFeedbackPage(page);
    await feedback.goto(sessionId);
    await feedback.open();
    await feedback.choose(2);
    const explanation = "Önemli bir şart açıklanmamış.";
    await feedback.fillExplanation(explanation);

    // Simulate only PUT /api/chat/chat-session/{sessionId}/feedback failure.
    const feedbackRoute = `**/api/chat/chat-session/${sessionId}/feedback`;
    await page.route(feedbackRoute, (route) =>
      route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Test failure" }),
      })
    );
    expect((await feedback.submit(sessionId)).status()).toBe(500);
    await feedback.expectError(/Değerlendirme kaydedilemedi/);
    await feedback.expectDraft(2, explanation);
    expect(await readFeedback(page.request, sessionId)).toBeNull();

    await page.unroute(feedbackRoute);
    await feedback.save(sessionId);
    expect(await readFeedback(page.request, sessionId)).toMatchObject({
      rating: 2,
      feedback_text: explanation,
      version: 1,
    });
  });

  test("version conflict refreshes the stored score while preserving the user's draft", async ({
    page,
  }) => {
    const feedback = new SessionFeedbackPage(page);
    await feedback.goto(sessionId);
    await feedback.open();
    await feedback.choose(1);
    const explanation = "Taslağımdaki yanlışlık açıklaması.";
    await feedback.fillExplanation(explanation);

    const externalUpdate = await page.request.put(
      `/api/chat/chat-session/${sessionId}/feedback`,
      {
        data: {
          rating: 4,
          feedback_text: "Başka istemcinin değerlendirmesi.",
          expected_version: 0,
        },
      }
    );
    expect(externalUpdate.status()).toBe(200);
    expect((await feedback.submit(sessionId)).status()).toBe(409);
    await feedback.expectError(/başka bir yerde güncellendi/);
    await feedback.expectDraft(1, explanation);
    await feedback.expectStoredRating(4);
    expect(await readFeedback(page.request, sessionId)).toMatchObject({
      rating: 4,
      version: 1,
    });

    await feedback.save(sessionId);
    expect(await readFeedback(page.request, sessionId)).toMatchObject({
      rating: 1,
      feedback_text: explanation,
      version: 2,
    });
  });
});
