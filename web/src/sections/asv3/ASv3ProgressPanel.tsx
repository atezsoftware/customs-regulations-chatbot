"use client";

import { Button, Text } from "@opal/components";
import { cn } from "@opal/utils";
import type { ASv3ProgressState } from "@/lib/asv3/progress";

interface ASv3ProgressPanelProps {
  state: ASv3ProgressState;
  stopped: boolean;
  onResume?: () => void;
  pending?: boolean;
}

/** Public task updates arrive already localized to the question language. */
export default function ASv3ProgressPanel({
  state,
  stopped,
  onResume,
  pending = false,
}: ASv3ProgressPanelProps) {
  if (!state.header && pending)
    return (
      <section
        aria-label="ASv3"
        aria-live="polite"
        aria-busy={!stopped}
        data-testid="asv3-progress"
        className="flex items-center gap-2 rounded-08 border border-border-02 bg-background-neutral-01 p-3"
      >
        <Text font="secondary-action" color="text-03">
          ASv3
        </Text>
        <span
          aria-hidden="true"
          className={cn(
            "h-1.5 w-1.5 rounded-full bg-background-neutral-04",
            !stopped && "animate-pulse"
          )}
        />
      </section>
    );
  if (!state.header) return null;
  const tasks = Array.from(state.tasks.values());
  return (
    <section
      aria-label="ASv3"
      aria-live="polite"
      aria-busy={!state.terminal && !stopped}
      lang={state.header.language}
      data-testid="asv3-progress"
      className="rounded-08 border border-border-02 bg-background-neutral-01 p-3"
    >
      <div className="flex items-center gap-2">
        <Text font="secondary-action" color="text-03">
          ASv3
        </Text>
        <Text font="main-ui-action" color="text-04">
          {state.header.title}
        </Text>
      </div>
      {state.header.message && (
        <div className="pt-1">
          <Text as="p" font="secondary-body" color="text-03">
            {state.header.message}
          </Text>
        </div>
      )}
      {tasks.length > 0 && (
        <ul className="flex flex-col gap-2 pt-3">
          {tasks.map((task) => (
            <li
              key={task.task_id}
              data-testid="asv3-task"
              data-task-id={task.task_id ?? undefined}
              data-status={task.status}
              className="flex items-start gap-2"
            >
              <span
                aria-hidden="true"
                className={cn(
                  "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full",
                  task.status === "failed"
                    ? "bg-status-error-03"
                    : task.status === "completed"
                      ? "bg-status-success-03"
                      : "bg-background-neutral-04",
                  !state.terminal &&
                    !stopped &&
                    task.status === "running" &&
                    "animate-pulse"
                )}
              />
              <div>
                <Text font="secondary-action" color="text-04">
                  {task.title}
                </Text>
                {task.message && (
                  <Text as="p" font="secondary-body" color="text-03">
                    {task.message}
                  </Text>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      {onResume &&
        state.header.phase === "interrupted" &&
        state.header.status === "failed" &&
        state.header.resume_label && (
          <div className="pt-3">
            <Button prominence="secondary" size="sm" onClick={onResume}>
              {state.header.resume_label}
            </Button>
          </div>
        )}
    </section>
  );
}
