import { Text } from "@opal/components";
import type { ChatSessionMinimal } from "@/app/ee/admin/performance/usage/types";

interface QueryHistoryUserProps {
  session: Pick<
    ChatSessionMinimal,
    | "user_name"
    | "user_email"
    | "api_key_id"
    | "api_key_name"
    | "api_key_display"
  >;
  anonymousLabel?: string;
}

export default function QueryHistoryUser({
  session,
  anonymousLabel = "-",
}: QueryHistoryUserProps) {
  const apiKeyLabel =
    session.api_key_name ||
    (session.api_key_id != null ? `API Key #${session.api_key_id}` : null);
  const apiKeyDetails = [apiKeyLabel, session.api_key_display]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="flex flex-col gap-1 break-words">
      <Text font="main-ui-body">
        {session.user_name || session.user_email || anonymousLabel}
      </Text>
      {apiKeyDetails && (
        <Text font="secondary-body" color="text-03">
          {apiKeyDetails}
        </Text>
      )}
    </div>
  );
}
