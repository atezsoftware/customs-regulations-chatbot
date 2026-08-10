"use client";

import { Tabs } from "@opal/components";
import SvgBarChart from "@opal/icons/bar-chart";
import { SettingsLayouts } from "@opal/layouts";

import BenchmarkQuestionsPanel from "@/views/admin/benchmark/BenchmarkQuestionsPanel";
import BenchmarkRunsPanel from "@/views/admin/benchmark/BenchmarkRunsPanel";

export default function BenchmarkPage() {
  return (
    <SettingsLayouts.Root width="full">
      <SettingsLayouts.Header
        icon={SvgBarChart}
        title="Benchmark"
        description="Build reproducible regulatory QA evaluations, run the exact production chat flow through OpenRouter models, and inspect evidence-grounded judge reports."
        divider
      />
      <SettingsLayouts.Body>
        <Tabs defaultValue="questions" variant="pill">
          <Tabs.List>
            <Tabs.Trigger value="questions">Questions</Tabs.Trigger>
            <Tabs.Trigger value="runs">Runs</Tabs.Trigger>
          </Tabs.List>
          <Tabs.Content value="questions">
            <BenchmarkQuestionsPanel />
          </Tabs.Content>
          <Tabs.Content value="runs">
            <BenchmarkRunsPanel />
          </Tabs.Content>
        </Tabs>
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
