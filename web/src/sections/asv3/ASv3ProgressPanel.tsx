"use client";

import { useId, useState } from "react";
import { Button, Tabs, Text } from "@opal/components";
import {
  SvgBranch,
  SvgCheckCircle,
  SvgCircle,
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
import { StepContainer as OnyxStepContainer } from "@/app/app/message/messageComponents/timeline/StepContainer";
import "@/sections/asv3/styles.css";

interface ASv3ProgressPanelProps {
  state: ASv3ProgressState;
  stopped: boolean;
  onResume?: () => void;
  pending?: boolean;
  agent?: MinimalAgent;
  hasDisplayContent?: boolean;
  workflowLabel?: string;
}

interface TaskStatusIconProps {
  task: ASv3Progress;
  active: boolean;
}

function isPlaceholder(event: ASv3Progress): boolean {
  const message = event.message?.trim();
  return event.title === "ASv3" && (!message || message === "…");
}

function progressTitle(event: ASv3Progress): string {
  if (
    !/^(?:Özgün Kaynak|Original source)\s*\[\d+\]$/i.test(event.title.trim())
  ) {
    return event.title;
  }
  const language = event.language.split("-")[0]?.toLowerCase();
  if (language === "tr") return "Kaynaklar inceleniyor";
  if (language === "en") return "Reviewing sources";
  const message = event.message?.trim();
  return message && message !== "…" ? message : "ASv3";
}

function TaskStatusIcon({ task, active }: TaskStatusIconProps) {
  const loading =
    active && (task.status === "queued" || task.status === "running");
  const Icon = loading
    ? SvgLoader
    : task.status === "completed"
      ? SvgCheckCircle
      : task.status === "failed" || task.status === "cancelled"
        ? SvgXOctagon
        : SvgCircle;
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

interface ASv3PastStepProps {
  step: ASv3Progress;
  first: boolean;
  last: boolean;
  active: boolean;
}

function ASv3PastStep({ step, first, last, active }: ASv3PastStepProps) {
  const detailsId = useId();
  const [expanded, setExpanded] = useState(false);
  const title = progressTitle(step);
  const Icon =
    step.status === "completed"
      ? SvgCheckCircle
      : step.status === "failed" || step.status === "cancelled"
        ? SvgXOctagon
        : SvgCircle;
  return (
    <div data-testid="asv3-past-step">
      <OnyxStepContainer
        stepIcon={Icon}
        isFirstStep={first}
        isLastStep={last}
        collapsible={false}
        header={
          <div className="asv3-step-toggle min-w-0">
            <Button
              prominence="tertiary"
              size="md"
              width="full"
              icon={() => <TaskStatusIcon task={step} active={active} />}
              rightIcon={expanded ? SvgFold : SvgExpand}
              aria-expanded={expanded}
              aria-controls={detailsId}
              onClick={() => setExpanded(!expanded)}
              title={title}
            >
              {title}
            </Button>
          </div>
        }
      >
        {expanded && (
          <div id={detailsId} className="px-2 pb-2">
            {step.message && (
              <Text as="p" font="secondary-body" color="text-03">
                {step.message}
              </Text>
            )}
          </div>
        )}
      </OnyxStepContainer>
    </div>
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
          <Tabs.List aria-label={progressTitle(selectedTask)}>
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
                    {progressTitle(task)}
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
  workflowLabel = "ASv3",
}: ASv3ProgressPanelProps) {
  const detailsId = useId();
  const [expanded, setExpanded] = useState(false);
  if (!state.header && (!pending || hasDisplayContent)) return null;
  const active = !state.terminal && !stopped && !hasDisplayContent;
  const allTasks = Array.from(state.tasks.values());
  if (
    hasDisplayContent &&
    state.header &&
    isPlaceholder(state.header) &&
    [...state.history, ...allTasks].every(isPlaceholder)
  ) {
    return null;
  }
  const actions = allTasks.filter((task) =>
    task.task_id?.startsWith("action:")
  );
  const tasks = allTasks.filter((task) => !task.task_id?.startsWith("action:"));
  const ids = new Set(tasks.map((task) => task.task_id));
  const roots = tasks.filter(
    (task) =>
      !task.parent_task_id ||
      !ids.has(task.parent_task_id) ||
      task.parent_task_id === task.task_id
  );
  const rootTasks = roots.length > 0 ? roots : tasks;
  const header = state.header;
  const history = state.history.filter(
    (step) =>
      !isPlaceholder(step) &&
      !actions.some(
        (action) =>
          action.title === step.title &&
          action.message === step.message &&
          action.phase === step.phase
      )
  );
  const pastSteps = [...history, ...actions]
    .filter((step) => step.event_id !== header?.event_id)
    .sort((left, right) => left.sequence - right.sequence);
  const title = header ? progressTitle(header) : workflowLabel;
  const hasHeaderMessage = Boolean(
    header?.message?.trim() && header.message.trim() !== "…"
  );
  const hasDetails = Boolean(
    hasHeaderMessage || pastSteps.length > 0 || tasks.length > 0
  );
  return (
    <section
      aria-label={workflowLabel}
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
          <div
            data-testid="asv3-progress-title"
            className={cn(
              "asv3-header-toggle flex h-full min-w-0 items-center p-1",
              active && "asv3-title-running"
            )}
          >
            {hasDetails ? (
              <Button
                prominence="tertiary"
                size="md"
                width="full"
                rightIcon={expanded ? SvgFold : SvgExpand}
                aria-expanded={expanded}
                aria-controls={detailsId}
                onClick={() => setExpanded(!expanded)}
                title={title}
              >
                {title}
              </Button>
            ) : (
              <div className="asv3-title-label px-2 py-1">
                <Text font="main-ui-action" color="text-03">
                  {title}
                </Text>
              </div>
            )}
          </div>
        </TimelineHeaderRow>
        {expanded && hasDetails && (
          <div
            id={detailsId}
            className={cn(
              active &&
                "motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-2 motion-safe:duration-300"
            )}
          >
            {hasHeaderMessage && header && (
              <TimelineRow
                railVariant="spacer"
                showIcon={false}
                isLast={pastSteps.length === 0 && tasks.length === 0}
              >
                <div className="px-3 pb-2">
                  <Text as="p" font="secondary-body" color="text-03">
                    {header.message ?? undefined}
                  </Text>
                </div>
              </TimelineRow>
            )}
            {pastSteps.map((step, index) => (
              <ASv3PastStep
                key={step.event_id}
                step={step}
                first={index === 0}
                last={index === pastSteps.length - 1 && tasks.length === 0}
                active={active}
              />
            ))}
            {tasks.length > 0 && (
              <ASv3TaskGroup
                tasks={rootTasks}
                allTasks={tasks}
                active={active}
              />
            )}
          </div>
        )}
        {onResume &&
          header?.workflow !== "supersearch" &&
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
