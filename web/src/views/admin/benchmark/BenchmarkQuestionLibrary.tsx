"use client";

import { Button, Card, Tag, Text } from "@opal/components";
import { toast } from "@opal/layouts";

import MinimalMarkdown from "@/components/chat/MinimalMarkdown";
import {
  type BenchmarkQuestion,
  deleteBenchmarkQuestion,
  updateBenchmarkQuestion,
} from "@/lib/regulatory/benchmark";

interface BenchmarkQuestionLibraryProps {
  questions: BenchmarkQuestion[];
  loading: boolean;
  onEdit: (question: BenchmarkQuestion) => void;
  onRefresh: () => Promise<void>;
}

export default function BenchmarkQuestionLibrary({
  questions,
  loading,
  onEdit,
  onRefresh,
}: BenchmarkQuestionLibraryProps) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <Text as="h2" font="heading-h3" color="text-05">
          Question library
        </Text>
        <Text font="main-ui-muted" color="text-02">
          {`${questions.length} questions`}
        </Text>
      </div>
      {loading && (
        <Card border="solid" padding="lg">
          <Text font="main-ui-muted" color="text-02">
            Loading questions…
          </Text>
        </Card>
      )}
      {!loading && questions.length === 0 && (
        <Card border="solid" padding="lg">
          <Text font="main-ui-muted" color="text-02">
            No benchmark questions yet.
          </Text>
        </Card>
      )}
      {questions.map((question) => (
        <Card key={question.id} border="solid" padding="lg">
          <details>
            <summary className="cursor-pointer list-none">
              <div className="flex items-start justify-between gap-4">
                <div className="flex min-w-0 flex-col gap-1">
                  <Text font="main-ui-action" color="text-05">
                    {question.title}
                  </Text>
                  <Text font="main-ui-body" color="text-03" maxLines={2}>
                    {question.prompt}
                  </Text>
                  <Text font="secondary-body" color="text-02">
                    {`${question.document_set_name}${question.as_of_date ? ` · as of ${question.as_of_date}` : ""} · ${question.expected_facts.length} facts · ${question.expected_citations.length} citations`}
                  </Text>
                </div>
                <div className="flex flex-wrap justify-end gap-1.5">
                  <Tag title={question.is_active ? "active" : "inactive"} />
                  {question.tags.slice(0, 4).map((tag) => (
                    <Tag key={tag} title={tag} />
                  ))}
                </div>
              </div>
            </summary>
            <div className="mt-4 border-t border-border-01 pt-4">
              {question.reference_answer && (
                <div className="mb-4">
                  <Text as="h3" font="secondary-action" color="text-02">
                    Gold answer
                  </Text>
                  <MinimalMarkdown content={question.reference_answer} />
                </div>
              )}
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <Text as="h3" font="secondary-action" color="text-02">
                    Expected facts
                  </Text>
                  <ul className="list-disc pl-5">
                    {question.expected_facts.map((fact) => (
                      <Text
                        as="li"
                        key={fact}
                        font="main-ui-body"
                        color="text-03"
                      >
                        {fact}
                      </Text>
                    ))}
                  </ul>
                </div>
                <div>
                  <Text as="h3" font="secondary-action" color="text-02">
                    Expected citations
                  </Text>
                  <ul className="list-disc pl-5">
                    {question.expected_citations.map((citation) => (
                      <Text
                        as="li"
                        key={citation.chunk_id}
                        font="main-ui-body"
                        color="text-03"
                      >
                        {`${citation.file_name} · ${citation.heading_path.join(" › ")} (${citation.requirement})`}
                      </Text>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <Button
                  size="sm"
                  prominence="secondary"
                  onClick={() => onEdit(question)}
                >
                  Edit
                </Button>
                <Button
                  size="sm"
                  prominence="secondary"
                  onClick={() =>
                    void updateBenchmarkQuestion(question.id, {
                      is_active: !question.is_active,
                    }).then(onRefresh)
                  }
                >
                  {question.is_active ? "Deactivate" : "Activate"}
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  prominence="secondary"
                  onClick={() =>
                    void deleteBenchmarkQuestion(question.id)
                      .then(onRefresh)
                      .catch((error: unknown) =>
                        toast.error(
                          error instanceof Error
                            ? error.message
                            : "Delete failed."
                        )
                      )
                  }
                >
                  Delete
                </Button>
              </div>
            </div>
          </details>
        </Card>
      ))}
    </section>
  );
}
