import { fireEvent, render, screen, within } from "@tests/setup/test-utils";
import ASv3ProgressPanel from "@/sections/asv3/ASv3ProgressPanel";
import {
  applyASv3Progress,
  createASv3ProgressState,
} from "@/lib/asv3/progress";
import type { ASv3Progress } from "@/app/app/services/streamingModels";

function parallelState() {
  let state = createASv3ProgressState();
  const events: ASv3Progress[] = [
    {
      type: "asv3_progress",
      run_id: "r",
      event_id: "1",
      sequence: 1,
      language: "tr",
      phase: "research",
      status: "running",
      title: "Garanti koşullarını inceliyorum",
    },
    {
      type: "asv3_progress",
      run_id: "r",
      event_id: "2",
      sequence: 2,
      language: "tr",
      phase: "worker",
      status: "completed",
      task_id: "repair",
      title: "Ücretsiz tamir",
      message: "Aynı makinenin geri geliş şartları incelendi.",
    },
    {
      type: "asv3_progress",
      run_id: "r",
      event_id: "3",
      sequence: 3,
      language: "tr",
      phase: "worker",
      status: "running",
      task_id: "replacement",
      title: "Yeni makine",
      message: "Farklı seri numaralı eşyanın ithalat koşulları inceleniyor.",
    },
  ];
  for (const event of events) state = applyASv3Progress(state, event);
  return state;
}

function completedState() {
  return applyASv3Progress(parallelState(), {
    type: "asv3_progress",
    run_id: "r",
    event_id: "done",
    sequence: 9,
    language: "tr",
    phase: "completed",
    status: "completed",
    title: "Araştırma tamamlandı",
  });
}

it("keeps sibling activity independent and changes the natural task content through Onyx tabs", () => {
  const state = parallelState();
  render(<ASv3ProgressPanel state={state} stopped={false} />);
  expect(screen.getAllByRole("tab")).toHaveLength(2);
  expect(
    within(screen.getByRole("tab", { name: "Ücretsiz tamir" })).queryByTestId(
      "asv3-task-loading"
    )
  ).not.toBeInTheDocument();
  expect(
    within(screen.getByRole("tab", { name: "Yeni makine" })).getByTestId(
      "asv3-task-loading"
    )
  ).toBeInTheDocument();
  fireEvent.mouseDown(screen.getByRole("tab", { name: "Yeni makine" }), {
    button: 0,
    ctrlKey: false,
  });
  expect(
    screen.getByText(
      "Farklı seri numaralı eşyanın ithalat koşulları inceleniyor."
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByText("Aynı makinenin geri geliş şartları incelendi.")
  ).not.toBeInTheDocument();
});

it("collapses at answer arrival and terminal replay without restarting activity", () => {
  const { rerender, unmount } = render(
    <ASv3ProgressPanel state={parallelState()} stopped={false} />
  );
  rerender(
    <ASv3ProgressPanel
      state={parallelState()}
      stopped={false}
      hasDisplayContent
    />
  );
  expect(screen.queryByRole("tab")).not.toBeInTheDocument();
  rerender(<ASv3ProgressPanel state={completedState()} stopped />);
  expect(
    screen.getByRole("button", { name: "Araştırma tamamlandı" })
  ).toHaveAttribute("aria-expanded", "false");
  expect(
    screen.getByText("Araştırma tamamlandı").closest(".shimmer-text")
  ).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Araştırma tamamlandı" }));
  expect(screen.getAllByRole("tab")).toHaveLength(2);
  expect(screen.queryByTestId("asv3-task-loading")).not.toBeInTheDocument();
  unmount();
  render(<ASv3ProgressPanel state={completedState()} stopped />);
  expect(screen.queryByRole("tab")).not.toBeInTheDocument();
  expect(screen.getByRole("region", { name: "ASv3" })).toHaveAttribute(
    "aria-busy",
    "false"
  );
});

it("respects manual expansion across final and does not reactivate a stopped sibling", () => {
  const { rerender } = render(
    <ASv3ProgressPanel state={parallelState()} stopped={false} />
  );
  const toggle = screen.getByRole("button", {
    name: "Garanti koşullarını inceliyorum",
  });
  fireEvent.click(toggle);
  fireEvent.click(toggle);
  rerender(
    <ASv3ProgressPanel state={parallelState()} stopped hasDisplayContent />
  );
  expect(screen.getAllByRole("tab")).toHaveLength(2);
  expect(screen.queryByTestId("asv3-task-loading")).not.toBeInTheDocument();
  rerender(
    <ASv3ProgressPanel state={completedState()} stopped hasDisplayContent />
  );
  expect(screen.getAllByRole("tab")).toHaveLength(2);
  expect(
    screen.getByRole("button", { name: "Araştırma tamamlandı" })
  ).toHaveAttribute("aria-expanded", "true");
});

it("groups a nested researcher beneath its selected parent without exposing IDs", () => {
  const state = applyASv3Progress(parallelState(), {
    type: "asv3_progress",
    run_id: "r",
    event_id: "nested",
    sequence: 4,
    language: "tr",
    phase: "worker",
    status: "running",
    task_id: "old-tax",
    parent_task_id: "replacement",
    title: "İlk ithalat vergileri",
    message: "Geri verme başvurusunun süresi kontrol ediliyor.",
  });
  render(<ASv3ProgressPanel state={state} stopped={false} />);
  expect(
    screen.queryByRole("tab", { name: "İlk ithalat vergileri" })
  ).not.toBeInTheDocument();
  fireEvent.mouseDown(screen.getByRole("tab", { name: "Yeni makine" }), {
    button: 0,
    ctrlKey: false,
  });
  expect(
    screen.getByRole("tab", { name: "İlk ithalat vergileri" })
  ).toBeInTheDocument();
  expect(
    screen.getByText("Geri verme başvurusunun süresi kontrol ediliyor.")
  ).toBeInTheDocument();
  expect(screen.queryByText("old-tax")).not.toBeInTheDocument();
});

it("shows a language-neutral pending state until the localized update arrives", () => {
  const empty = createASv3ProgressState();
  const { rerender } = render(
    <ASv3ProgressPanel state={empty} stopped={false} pending />
  );
  expect(screen.getByRole("region", { name: "ASv3" })).toHaveTextContent(
    /^ASv3$/
  );
  expect(screen.queryByText(/Thinking/)).not.toBeInTheDocument();
  const localized = applyASv3Progress(empty, {
    type: "asv3_progress",
    run_id: "r",
    event_id: "e",
    sequence: 1,
    language: "tr",
    phase: "started",
    status: "running",
    title: "Garanti koşullarını inceliyorum",
  });
  rerender(<ASv3ProgressPanel state={localized} stopped={false} pending />);
  expect(
    screen.getByText("Garanti koşullarını inceliyorum")
  ).toBeInTheDocument();
  expect(screen.getByRole("region", { name: "ASv3" })).toHaveAttribute(
    "lang",
    "tr"
  );
});

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
