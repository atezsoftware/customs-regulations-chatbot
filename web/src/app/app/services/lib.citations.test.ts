import { BackendMessage } from "@/app/app/interfaces";
import { getCitations } from "@/app/app/services/packetUtils";
import { processRawChatHistory } from "@/app/app/services/lib";
import { Packet } from "@/app/app/services/streamingModels";

const savedAnswer: BackendMessage = {
  message_id: 2,
  message_type: "assistant",
  research_type: null,
  parent_message: 1,
  latest_child_message: null,
  message: "Yanıt [[10]]() ve [[11]]()",
  rephrased_query: null,
  context_docs: [],
  time_sent: "2026-10-22T10:00:00Z",
  overridden_model: "gpt-6.1-sol",
  alternate_assistant_id: null,
  chat_session_id: "session-1",
  citations: { 10: "same-file", 11: "same-file" },
  files: [],
  tool_call: null,
  current_feedback: null,
  sub_questions: [],
  comments: null,
  parentMessageId: 1,
  refined_answer_improvement: null,
  is_agentic: null,
  preferred_response_id: null,
  model_display_name: "Sol 6.1",
  error: null,
};

it("keeps old saved packets intact when no detailed citation array is available", () => {
  const packets: Packet[] = [
    {
      placement: { turn_index: 0 },
      obj: {
        type: "citation_info",
        citation_number: 10,
        document_id: "same-file",
        chunk_ind: 46,
      },
    },
  ];
  const result = processRawChatHistory([savedAnswer], [packets]).get(2);
  expect(result?.packets).toBe(packets);
  expect(result?.citations).toEqual(savedAnswer.citations);
  expect(result?.message).toBe(savedAnswer.message);
});

it("restores each saved citation's official title and exact chunk independently", () => {
  const answer: BackendMessage = {
    ...savedAnswer,
    citation_info: [46, 47].map((chunk, index) => ({
      type: "citation_info",
      citation_number: 10 + index,
      document_id: "same-file",
      chunk_ind: chunk,
      chunk_id: `canonical-${chunk}`,
      display_title: "Resmî Başlık",
      text: `Madde ${chunk} özgün metni`,
    })),
  };
  const result = processRawChatHistory([answer], [[]]).get(2);
  expect(getCitations(result?.packets || [])).toEqual(
    answer.citation_info?.map(({ type: _type, citation_number, ...citation }) =>
      expect.objectContaining({
        ...citation,
        citation_num: citation_number,
      })
    )
  );
  expect(result?.citations).toEqual(savedAnswer.citations);
  expect(result?.message).toBe(savedAnswer.message);
});
