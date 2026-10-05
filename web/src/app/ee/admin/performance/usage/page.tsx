"use client";

import { AdminDateRangeSelector } from "@/components/dateRangeSelectors/AdminDateRangeSelector";
import { OnyxBotChart } from "@/app/ee/admin/performance/usage/OnyxBotChart";
import { FeedbackChart } from "@/app/ee/admin/performance/usage/FeedbackChart";
import { QueryPerformanceChart } from "@/app/ee/admin/performance/usage/QueryPerformanceChart";
import { PersonaMessagesChart } from "@/app/ee/admin/performance/usage/PersonaMessagesChart";
import { useTimeRange } from "@/app/ee/admin/performance/lib";
import UsageReports from "@/app/ee/admin/performance/usage/UsageReports";
import PerUserUsagePanel from "@/views/admin/PerUserUsagePanel";
import { useState } from "react";
import useSWR from "swr";
import { errorHandlingFetcher } from "@/lib/fetcher";
import { Button, Text } from "@opal/components";
import { Divider } from "@opal/components";
import { useAdminAgents } from "@/lib/agents/hooks";
import { ADMIN_ROUTES } from "@/lib/admin-routes";
import { SettingsLayouts } from "@opal/layouts";
import UsageSummarySection from "@/app/ee/admin/performance/usage/UsageSummary";

const route = ADMIN_ROUTES.USAGE;

export default function AnalyticsPage() {
  const [workflow, setWorkflow] = useState<string | undefined>();
  const { data: measurement } = useSWR<{
    id: string;
    started_at: string;
  } | null>("/api/admin/usage/measurement-period", errorHandlingFetcher);
  const [timeRange, setTimeRange] = useTimeRange();
  const { agents } = useAdminAgents();

  return (
    <SettingsLayouts.Root>
      <SettingsLayouts.Header icon={route.icon} title={route.title} divider />
      <SettingsLayouts.Body>
        <AdminDateRangeSelector
          value={timeRange}
          onValueChange={(value) => setTimeRange(value as any)}
        />
        {measurement && (
          <Text as="p">
            {`Usage measurement started ${new Date(measurement.started_at).toLocaleString()}.`}
          </Text>
        )}
        <div className="flex gap-2">
          {[
            { label: "All workflows", value: undefined },
            { label: "Normal", value: "normal" },
            { label: "Deep Research", value: "deep" },
          ].map((option) => (
            <Button
              key={option.label}
              prominence={workflow === option.value ? "primary" : "secondary"}
              onClick={() => setWorkflow(option.value)}
            >
              {option.label}
            </Button>
          ))}
        </div>
        <UsageSummarySection timeRange={timeRange} workflow={workflow} />
        <QueryPerformanceChart timeRange={timeRange} />
        <FeedbackChart timeRange={timeRange} />
        <OnyxBotChart timeRange={timeRange} />
        <PersonaMessagesChart
          availablePersonas={agents}
          timeRange={timeRange}
        />
        <Divider />
        <PerUserUsagePanel timeRange={timeRange} workflow={workflow} />
        <Divider />
        <UsageReports />
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
