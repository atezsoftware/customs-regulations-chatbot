import { test, expect } from "@tests/e2e/chat/fixtures";
import type { Packet, ASv3Progress } from "@/app/app/services/streamingModels";

// This checks the browser contract against a deterministic ASv3 stream.
// Live harness quality and persisted-history acceptance are separate backend tests.
for (const scenario of [
  {
    language: "tr",
    native: false,
    question:
      "Garanti kapsamında tamir ve yeni makine aynı muafiyeti sağlar mı?",
    title: "Garanti kapsamındaki iki ihtimali inceliyorum",
    tasks: [
      "Bedelsiz tamir şartları değerlendirildi",
      "Yeni makinenin ithalat koşulları değerlendirildi",
    ],
    answer: "İki işlem aynı muafiyet sonucunu doğurmaz.",
    done: "Araştırma tamamlandı",
  },
  {
    language: "en",
    native: true,
    question:
      "Do a free warranty repair and a replacement machine have the same exemption?",
    title: "I am examining the two warranty scenarios",
    tasks: [
      "Reviewed the conditions for free repair",
      "Reviewed the import conditions for a replacement",
    ],
    answer: "The two transactions do not have the same exemption.",
    done: "Research completed",
  },
]) {
  test(`ASv3 selection, localized parallel updates and citation target (${scenario.language})`, async ({
    chatPage,
  }) => {
    const doc = {
      document_id: scenario.native ? "original-policy-pdf" : "asv3-test-law",
      chunk_ind: scenario.native ? -1 : 7,
      semantic_identifier: scenario.native
        ? "Warranty policy.pdf · page 4 (original source excerpt)"
        : "Gümrük Kanunu · 142",
      link: "",
      source_type: "user_file",
      blurb: scenario.answer,
      is_internet: false,
      boost: 0,
      hidden: false,
      score: 1,
      match_highlights: [],
      metadata: {},
      updated_at: null,
    };
    const event = (
      sequence: number,
      changes: Partial<ASv3Progress> = {}
    ): Packet => ({
      placement: { turn_index: 0 },
      obj: {
        type: "asv3_progress",
        run_id: "mock-asv3",
        event_id: `event-${sequence}`,
        sequence,
        language: scenario.language,
        phase: "research",
        status: "running",
        title: scenario.title,
        ...changes,
      },
    });
    const packets: unknown[] = [
      { user_message_id: 101, reserved_assistant_message_id: 102 },
      event(1),
      event(2, {
        task_id: "repair",
        title: scenario.tasks[0],
        status: "running",
      }),
      event(3, {
        task_id: "replacement",
        title: scenario.tasks[1],
        status: "running",
      }),
      { placement: { turn_index: 1 }, obj: { type: "search_tool_start" } },
      {
        placement: { turn_index: 1 },
        obj: { type: "search_tool_documents_delta", documents: [doc] },
      },
      event(4, {
        task_id: "repair",
        title: scenario.tasks[0],
        status: "completed",
      }),
      event(5, {
        task_id: "replacement",
        title: scenario.tasks[1],
        status: "completed",
      }),
      event(6, {
        phase: "completed",
        title: scenario.done,
        status: "completed",
      }),
      {
        placement: { turn_index: 2 },
        obj: {
          type: "message_start",
          id: "mock-102",
          content: "",
          final_documents: [doc],
        },
      },
      {
        placement: { turn_index: 2 },
        obj: {
          type: "citation_info",
          citation_number: 1,
          document_id: doc.document_id,
          chunk_ind: doc.chunk_ind,
          semantic_identifier: doc.semantic_identifier,
          source_type: doc.source_type,
          ...(scenario.native
            ? { preview_url: "/api/asv3/citation/102/1" }
            : {}),
        },
      },
      {
        placement: { turn_index: 2 },
        obj: { type: "message_delta", content: `${scenario.answer} [[1]]()` },
      },
      { placement: { turn_index: 2 }, obj: { type: "section_end" } },
      {
        placement: { turn_index: 2 },
        obj: { type: "stop", stop_reason: "finished" },
      },
      { message_id: 102, citations: { 1: doc.document_id }, files: [] },
    ];
    const requests: Record<string, unknown>[] = [];
    // POST /api/chat/send-chat-message: deterministic progress + final answer.
    await chatPage.page.route(
      "**/api/chat/send-chat-message",
      async (route) => {
        requests.push(
          route.request().postDataJSON() as Record<string, unknown>
        );
        await expect(chatPage.page.getByTestId("asv3-progress")).toHaveText(
          "ASv3"
        );
        await expect(
          chatPage.page.getByText("Thinking...", { exact: true })
        ).toHaveCount(0);
        await route.fulfill({
          status: 200,
          contentType: "text/plain",
          body:
            packets.map((packet) => JSON.stringify(packet)).join("\n") + "\n",
        });
      }
    );
    // GET /api/document/chunk-info: the exact citation target, never the full file.
    await chatPage.page.route(
      "**/api/document/chunk-info?**",
      async (route) => {
        await route.fulfill({
          status: 200,
          json: { content: scenario.answer },
        });
      }
    );
    await chatPage.page.route("**/api/asv3/citation/102/1", async (route) => {
      await route.fulfill({
        status: 200,
        json: { content: scenario.answer, num_tokens: 10 },
      });
    });
    await chatPage.goto();
    await chatPage.inputBar.selectASv3();
    await chatPage.inputBar.fill(scenario.question);
    await chatPage.inputBar.send();
    await chatPage.expectHumanMessage(scenario.question);
    await chatPage.expectASv3Progress(
      scenario.language,
      scenario.done,
      scenario.tasks
    );
    await chatPage.expectASv3TaskStatus("repair", "completed");
    await chatPage.expectASv3TaskStatus("replacement", "completed");
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      atez_search_v3: true,
      asv3_allow_external: false,
      atez_search: false,
      atez_search_v2: false,
      deep_research: false,
    });
    await chatPage.expectASv3CitationTarget(
      scenario.answer,
      doc.document_id,
      doc.chunk_ind,
      scenario.native ? "/api/asv3/citation/102/1" : undefined
    );
  });
}
