"use client";

import { AdminDateRangeSelector } from "@/components/dateRangeSelectors/AdminDateRangeSelector";
import { OnyxBotChart } from "@/app/ee/admin/performance/usage/OnyxBotChart";
import { FeedbackChart } from "@/app/ee/admin/performance/usage/FeedbackChart";
import { QueryPerformanceChart } from "@/app/ee/admin/performance/usage/QueryPerformanceChart";
import { PersonaMessagesChart } from "@/app/ee/admin/performance/usage/PersonaMessagesChart";
import { useTimeRange } from "@/app/ee/admin/performance/lib";
import UsageReports from "@/app/ee/admin/performance/usage/UsageReports";
import PerUserUsagePanel from "@/views/admin/PerUserUsagePanel";
import { Divider } from "@opal/components";
import { useAdminAgents } from "@/lib/agents/hooks";
import { ADMIN_ROUTES } from "@/lib/admin-routes";
import { SettingsLayouts } from "@opal/layouts";
import UsageSummarySection from "./UsageSummary";

const route = ADMIN_ROUTES.USAGE;

export default function AnalyticsPage() {
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
        <UsageSummarySection timeRange={timeRange} />
        <QueryPerformanceChart timeRange={timeRange} />
        <FeedbackChart timeRange={timeRange} />
        <OnyxBotChart timeRange={timeRange} />
        <PersonaMessagesChart
          availablePersonas={agents}
          timeRange={timeRange}
        />
        <Divider />
        <PerUserUsagePanel timeRange={timeRange} />
        <Divider />
        <UsageReports />
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
