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

const totalCostFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const averageCostFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 4,
  maximumFractionDigits: 4,
});

function MetricCard({
  label,
  value,
  details,
}: {
  label: string;
  value: string;
  details: string[];
}) {
  return (
    <div className="rounded-12 border bg-background-neutral-01 p-4">
      <Text as="p" color="text-03">
        {label}
      </Text>
      <Text as="p" font="heading-h2">
        {value}
      </Text>
      {details.map((detail) => (
        <Text as="p" color="text-03" key={detail}>
          {detail}
        </Text>
      ))}
    </div>
  );
}

export function UsageSummaryCards({ summary }: { summary: UsageSummary }) {
  return (
    <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        label="Total user queries"
        value={numberFormatter.format(summary.total_user_queries)}
        details={[
          `${numberFormatter.format(summary.average_tokens_per_query)} tokens / query`,
          `${averageCostFormatter.format(summary.average_cost_cents_per_query / 100)} / query`,
        ]}
      />
      <MetricCard
        label="Total user sessions"
        value={numberFormatter.format(summary.total_user_sessions)}
        details={[
          `${numberFormatter.format(summary.average_tokens_per_session)} tokens / session`,
          `${averageCostFormatter.format(summary.average_cost_cents_per_session / 100)} / session`,
          `${numberFormatter.format(summary.average_queries_per_session)} queries / session`,
        ]}
      />
      <MetricCard
        label="Total tokens"
        value={numberFormatter.format(summary.total_tokens)}
        details={["Input + output model tokens"]}
      />
      <MetricCard
        label="Total cost"
        value={totalCostFormatter.format(summary.total_cost_cents / 100)}
        details={["Tracked model usage"]}
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
        Queries, sessions, total model usage, and cost rates for the selected
        period.
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
