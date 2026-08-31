"use client";

import CardSection from "@/components/admin/CardSection";
import { DateRangePickerValue } from "@/components/dateRangeSelectors/AdminDateRangeSelector";
import { ErrorCallout } from "@/components/ErrorCallout";
import Title from "@/components/ui/title";
import { useUsageSummary } from "../lib";
import { UsageSummary } from "./types";
import { Text } from "@opal/components";
import SvgSimpleLoader from "@opal/icons/simple-loader";

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});

function MetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: number;
  detail: string;
}) {
  return (
    <div className="rounded-12 border bg-background-neutral-01 p-4">
      <Text as="p" color="text-03">
        {label}
      </Text>
      <Text as="p" font="heading-h2">
        {numberFormatter.format(value)}
      </Text>
      <Text as="p" color="text-03">
        {detail}
      </Text>
    </div>
  );
}

export function UsageSummaryCards({ summary }: { summary: UsageSummary }) {
  return (
    <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
      <MetricCard
        label="Total user queries"
        value={summary.total_user_queries}
        detail={`${numberFormatter.format(summary.average_tokens_per_query)} tokens / query`}
      />
      <MetricCard
        label="Total user sessions"
        value={summary.total_user_sessions}
        detail={`${numberFormatter.format(summary.average_queries_per_session)} queries / session`}
      />
      <MetricCard
        label="Total query tokens"
        value={summary.total_query_tokens}
        detail={`${numberFormatter.format(summary.average_tokens_per_session)} tokens / session`}
      />
    </div>
  );
}

export default function UsageSummarySection({
  timeRange,
}: {
  timeRange: DateRangePickerValue;
}) {
  const { data, error, isLoading } = useUsageSummary(timeRange);

  return (
    <CardSection className="mt-8">
      <Title>User activity totals</Title>
      <Text as="p">
        Queries, sessions, and token rates for the selected period.
      </Text>
      {isLoading ? (
        <div className="flex h-32 items-center justify-center">
          <SvgSimpleLoader className="h-6 w-6" />
        </div>
      ) : error || !data ? (
        <ErrorCallout
          errorTitle="Failed to load usage totals."
          errorMsg={
            (error as Error | undefined)?.message ?? "No data returned."
          }
        />
      ) : (
        <UsageSummaryCards summary={data} />
      )}
    </CardSection>
  );
}
