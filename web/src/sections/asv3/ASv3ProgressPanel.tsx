"use client";

import { useId, useState } from "react";
import { Button, Tabs, Text } from "@opal/components";
import {
  SvgBranch,
  SvgCheckCircle,
  SvgExpand,
  SvgFold,
  SvgLoader,
  SvgXOctagon,
} from "@opal/icons";
import { SvgOnyxLogo } from "@opal/logos";
import { cn } from "@opal/utils";
import type { ASv3Progress } from "@/app/app/services/streamingModels";
import type { MinimalAgent } from "@/lib/agents/types";
import type { ASv3ProgressState } from "@/lib/asv3/progress";
import AgentAvatar from "@/refresh-components/avatars/AgentAvatar";
import { TimelineRoot } from "@/app/app/message/messageComponents/timeline/primitives/TimelineRoot";
import { TimelineHeaderRow } from "@/app/app/message/messageComponents/timeline/primitives/TimelineHeaderRow";
import { TimelineRow } from "@/app/app/message/messageComponents/timeline/primitives/TimelineRow";
import { TimelineSurface } from "@/app/app/message/messageComponents/timeline/primitives/TimelineSurface";
import "@/sections/asv3/styles.css";

interface ASv3ProgressPanelProps {
  state: ASv3ProgressState;
  stopped: boolean;
  onResume?: () => void;
  pending?: boolean;
  agent?: MinimalAgent;
  hasDisplayContent?: boolean;
}

interface TaskStatusIconProps {
  task: ASv3Progress;
  active: boolean;
}

function TaskStatusIcon({ task, active }: TaskStatusIconProps) {
  const loading =
    active && (task.status === "queued" || task.status === "running");
  const Icon = loading
    ? SvgLoader
    : task.status === "completed"
      ? SvgCheckCircle
      : SvgXOctagon;
  return (
    <span
      aria-hidden="true"
      data-testid={loading ? "asv3-task-loading" : undefined}
    >
      <Icon
        size={14}
        className={cn(
          loading && "motion-safe:animate-spin",
          task.status === "completed" && "text-status-success-05",
          task.status === "failed" && "text-status-error-05",
          task.status === "cancelled" && "text-text-02"
        )}
      />
    </span>
  );
}

interface ASv3TaskGroupProps {
  tasks: ASv3Progress[];
  allTasks: ASv3Progress[];
  active: boolean;
  ancestry?: readonly string[];
}

/** Same rail, surfaces and parallel-tab navigation as the native Onyx timeline. */
function ASv3TaskGroup({
  tasks,
  allTasks,
  active,
  ancestry = [],
}: ASv3TaskGroupProps) {
  const [selected, setSelected] = useState(tasks[0]?.task_id ?? "");
  const selectedTask =
    tasks.find((task) => task.task_id === selected) ?? tasks[0];
  if (!selectedTask?.task_id) return null;
  const children = allTasks.filter(
    (task) =>
      task.parent_task_id === selectedTask.task_id &&
      task.task_id !== selectedTask.task_id &&
      !ancestry.includes(task.task_id ?? "")
  );
  return (
    <Tabs
      value={selectedTask.task_id}
      onValueChange={setSelected}
      variant="pill"
    >
      <TimelineRow
        icon={<SvgBranch size={16} />}
        isFirst
        isLast={children.length === 0}
      >
        <TimelineSurface
          roundedTop
          roundedBottom={children.length === 0}
          className="min-w-0 flex-1 p-1"
        >
          <Tabs.List aria-label={selectedTask.title}>
            {tasks.map((task) => (
              <Tabs.Trigger
                key={task.task_id}
                value={task.task_id ?? ""}
                data-testid="asv3-task"
                data-task-id={task.task_id ?? undefined}
                data-status={task.status}
              >
                <span className="flex items-center gap-1.5">
                  <TaskStatusIcon task={task} active={active} />
                  <Text font="secondary-action" color="inherit">
                    {task.title}
                  </Text>
                </span>
              </Tabs.Trigger>
            ))}
          </Tabs.List>
          {selectedTask.message && (
            <Tabs.Content value={selectedTask.task_id}>
              <div className="p-2">
                <Text as="p" font="secondary-body" color="text-03">
                  {selectedTask.message}
                </Text>
              </div>
            </Tabs.Content>
          )}
        </TimelineSurface>
      </TimelineRow>
      {children.length > 0 && ancestry.length < 3 && (
        <ASv3TaskGroup
          key={selectedTask.task_id}
          tasks={children}
          allTasks={allTasks}
          active={active}
          ancestry={[...ancestry, selectedTask.task_id]}
        />
      )}
    </Tabs>
  );
}

/** Public localized updates use Onyx presentation without generic tool-name renderers. */
export default function ASv3ProgressPanel({
  state,
  stopped,
  onResume,
  pending = false,
  agent,
  hasDisplayContent = false,
}: ASv3ProgressPanelProps) {
  const detailsId = useId();
  const [manualExpanded, setManualExpanded] = useState<boolean | null>(null);
  if (!state.header && !pending) return null;
  const active = !state.terminal && !stopped;
  const tasks = Array.from(state.tasks.values());
  const ids = new Set(tasks.map((task) => task.task_id));
  const roots = tasks.filter(
    (task) =>
      !task.parent_task_id ||
      !ids.has(task.parent_task_id) ||
      task.parent_task_id === task.task_id
  );
  const rootTasks = roots.length > 0 ? roots : tasks;
  const expanded = manualExpanded ?? (active && !hasDisplayContent);
  const header = state.header;
  const title = header?.title ?? "ASv3";
  return (
    <section
      aria-label="ASv3"
      aria-live="polite"
      aria-busy={active}
      lang={header?.language}
      data-testid="asv3-progress"
      className="asv3-timeline"
    >
      <TimelineRoot>
        <TimelineHeaderRow
          left={
            agent ? (
              <AgentAvatar agent={agent} size={24} />
            ) : (
              <SvgOnyxLogo size={24} />
            )
          }
        >
          <div className="flex h-full min-w-0 items-center justify-between rounded-12 p-1 transition-colors hover:bg-background-tint-00">
            <div
              data-testid="asv3-progress-title"
              className={cn(
                "min-w-0 px-(--timeline-header-text-padding-x) py-(--timeline-header-text-padding-y)",
                active && "shimmer-text"
              )}
            >
              <Text
                as="p"
                font="main-ui-action"
                color={active ? "inherit" : "text-03"}
                maxLines={1}
                title={title}
              >
                {title}
              </Text>
            </div>
            {tasks.length > 0 && (
              <Button
                prominence="tertiary"
                size="sm"
                icon={expanded ? SvgFold : SvgExpand}
                aria-label={title}
                aria-expanded={expanded}
                aria-controls={detailsId}
                onClick={() => setManualExpanded(!expanded)}
              />
            )}
          </div>
        </TimelineHeaderRow>
        {header?.message && (
          <TimelineRow
            railVariant="spacer"
            showIcon={false}
            isLast={!expanded || tasks.length === 0}
          >
            <div className="px-3 pb-2">
              <Text as="p" font="secondary-body" color="text-03">
                {header.message}
              </Text>
            </div>
          </TimelineRow>
        )}
        {expanded && tasks.length > 0 && (
          <div
            id={detailsId}
            className={cn(
              active &&
                "motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-2 motion-safe:duration-300"
            )}
          >
            <ASv3TaskGroup tasks={rootTasks} allTasks={tasks} active={active} />
          </div>
        )}
        {onResume &&
          header?.phase === "interrupted" &&
          header.status === "failed" &&
          header.resume_label && (
            <TimelineRow railVariant="spacer" showIcon={false} isLast>
              <div className="p-2">
                <Button prominence="secondary" size="sm" onClick={onResume}>
                  {header.resume_label}
                </Button>
              </div>
            </TimelineRow>
          )}
      </TimelineRoot>
    </section>
  );
}
