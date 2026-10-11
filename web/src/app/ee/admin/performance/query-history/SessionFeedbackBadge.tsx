import { Text } from "@opal/components";
import { SvgStar } from "@opal/icons";
import type { SessionFeedbackResponse } from "@/app/app/interfaces";

interface SessionFeedbackBadgeProps {
  feedback?: SessionFeedbackResponse | null;
}

export function SessionFeedbackBadge({ feedback }: SessionFeedbackBadgeProps) {
  if (!feedback) {
    return (
      <Text font="secondary-body" color="text-03">
        -
      </Text>
    );
  }

  return (
    <div
      className="flex flex-col gap-1"
      aria-label={`${feedback.rating}/4: ${feedback.label}`}
    >
      <div className="flex gap-0.5" aria-hidden="true">
        {[1, 2, 3, 4].map((rating) => (
          <SvgStar
            key={rating}
            size={12}
            fill={rating <= feedback.rating ? "currentColor" : "none"}
          />
        ))}
      </div>
      <Text font="secondary-body">{feedback.label}</Text>
    </div>
  );
}
