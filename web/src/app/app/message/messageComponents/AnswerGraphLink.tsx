"use client";

import { Button } from "@opal/components";
import { SvgWorkflow } from "@opal/icons";
import { useUser } from "@/providers/UserProvider";

interface AnswerGraphLinkProps {
  messageId?: number;
}

export default function AnswerGraphLink({ messageId }: AnswerGraphLinkProps) {
  const { isAdmin } = useUser();
  if (!isAdmin || messageId === undefined) return null;

  return (
    <Button
      icon={SvgWorkflow}
      size="sm"
      prominence="tertiary"
      tooltip="View this answer's execution graph in a new tab"
      data-testid="AgentMessage/answer-graph"
      onClick={(event) => {
        event.stopPropagation();
        window.open(
          "/admin/answer-graphs/message/" + messageId,
          "_blank",
          "noopener,noreferrer"
        );
      }}
    >
      Execution graph
    </Button>
  );
}
