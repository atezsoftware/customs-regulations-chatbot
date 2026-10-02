import { fireEvent, render, screen } from "@tests/setup/test-utils";
import ASv3ProgressPanel from "@/sections/asv3/ASv3ProgressPanel";
import {
  applyASv3Progress,
  createASv3ProgressState,
} from "@/lib/asv3/progress";
import type { ASv3Progress } from "@/app/app/services/streamingModels";

it.each([
  ["tr", "Kaynaklar paralel inceleniyor", "Süre koşulu araştırılıyor"],
  ["en", "Reviewing sources in parallel", "Checking the deadline"],
])(
  "shows backend progress in question language %s without generic English headers",
  (language, title, taskTitle) => {
    const initial: ASv3Progress = {
      type: "asv3_progress",
      run_id: "run-1",
      event_id: "1",
      sequence: 1,
      language,
      phase: "research",
      status: "running",
      title,
    };
    let state = applyASv3Progress(createASv3ProgressState(), initial);
    state = applyASv3Progress(state, {
      ...initial,
      event_id: "2",
      sequence: 2,
      task_id: "a",
      title: taskTitle,
    });
    const { rerender } = render(
      <ASv3ProgressPanel state={state} stopped={false} />
    );
    expect(screen.getByText(title)).toBeInTheDocument();
    expect(screen.getByText(taskTitle)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "ASv3" })).toHaveAttribute(
      "lang",
      language
    );
    expect(screen.getByRole("region", { name: "ASv3" })).toHaveAttribute(
      "aria-busy",
      "true"
    );
    expect(screen.queryByText("Research Task")).not.toBeInTheDocument();
    state = applyASv3Progress(state, {
      ...initial,
      event_id: "3",
      sequence: 3,
      status: "completed",
      title,
    });
    rerender(<ASv3ProgressPanel state={state} stopped={true} />);
    expect(screen.getByRole("region", { name: "ASv3" })).toHaveAttribute(
      "aria-busy",
      "false"
    );
  }
);

it("shows contextual public updates without tool names or private task identifiers", () => {
  const state = applyASv3Progress(createASv3ProgressState(), {
    type: "asv3_progress",
    run_id: "private-run",
    event_id: "internal-event",
    sequence: 1,
    language: "tr",
    phase: "read_provision",
    status: "running",
    title: "Garanti kapsamındaki iki ihtimali inceliyorum",
    message:
      "Gümrük Kanunu'ndaki bedelsiz tamir şartlarını yeni makine gönderilmesinden ayrı değerlendiriyorum.",
  });
  render(<ASv3ProgressPanel state={state} stopped={false} />);
  expect(
    screen.getByText("Garanti kapsamındaki iki ihtimali inceliyorum")
  ).toBeInTheDocument();
  expect(
    screen.getByText(/Gümrük Kanunu'ndaki bedelsiz tamir şartlarını/)
  ).toBeInTheDocument();
  for (const privateText of [
    "read_provision",
    "query_corpus",
    "spawn_researcher",
    "private-run",
    "internal-event",
  ]) {
    expect(screen.queryByText(privateText)).not.toBeInTheDocument();
  }
});

it("offers only a backend-localized explicit interrupted-run resume action", () => {
  const onResume = jest.fn();
  const initial: ASv3Progress = {
    type: "asv3_progress",
    run_id: "run-1",
    event_id: "resume",
    sequence: 5,
    language: "tr",
    phase: "interrupted",
    status: "failed",
    title: "Araştırma yarıda kaldı",
    resume_label: "Araştırmaya devam et",
  };
  const state = applyASv3Progress(createASv3ProgressState(), initial);
  const { rerender } = render(
    <ASv3ProgressPanel state={state} stopped onResume={onResume} />
  );
  fireEvent.click(screen.getByRole("button", { name: "Araştırmaya devam et" }));
  expect(onResume).toHaveBeenCalledTimes(1);
  const failed = applyASv3Progress(createASv3ProgressState(), {
    ...initial,
    phase: "research",
  });
  rerender(<ASv3ProgressPanel state={failed} stopped onResume={onResume} />);
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
});
