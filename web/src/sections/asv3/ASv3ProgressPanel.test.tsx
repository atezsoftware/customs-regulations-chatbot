import {
  fireEvent,
  render,
  screen,
  setupUser,
  within,
} from "@tests/setup/test-utils";
import ASv3ProgressPanel from "@/sections/asv3/ASv3ProgressPanel";
import {
  applyASv3Progress,
  createASv3ProgressState,
} from "@/lib/asv3/progress";
import type { ASv3Progress } from "@/app/app/services/streamingModels";

it.each([
  [
    "tr-TR",
    "Özgün Kaynak [9]",
    "Kaynaklar inceleniyor",
    "Özgün hükümler inceleniyor.",
  ],
  [
    "en-US",
    "Original source [27]",
    "Reviewing sources",
    "Reviewing original provisions.",
  ],
  [
    "de",
    "Original source [9]",
    "Die relevanten Bestimmungen werden geprüft.",
    "Die relevanten Bestimmungen werden geprüft.",
  ],
])(
  "replays opaque legacy source titles safely in %s without changing source identities",
  (language, legacyTitle, safeTitle, message) => {
    const original: ASv3Progress = {
      type: "asv3_progress",
      run_id: "legacy-run",
      event_id: "source-delivery",
      sequence: 1,
      language,
      phase: "tools",
      status: "completed",
      task_id: "action:source:original-identity",
      title: legacyTitle,
      message,
    };
    let state = applyASv3Progress(createASv3ProgressState(), original);
    state = applyASv3Progress(state, {
      ...original,
      event_id: "official-source",
      sequence: 2,
      task_id: "action:source:official-identity",
      title: "2024/22 Genelge",
    });
    state = applyASv3Progress(state, {
      ...original,
      event_id: "done",
      sequence: 3,
      task_id: null,
      title: "ASv3",
    });

    render(<ASv3ProgressPanel state={state} stopped />);
    fireEvent.click(screen.getByRole("button", { name: "ASv3" }));

    expect(screen.queryByText(legacyTitle)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: safeTitle })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "2024/22 Genelge" })
    ).toBeInTheDocument();
    expect(state.tasks.get(original.task_id!)).toBe(original);
    expect(original.title).toBe(legacyTitle);
  }
);

it("shows sequential source actions as distinct informative history, without fake parallel tabs or duplicate headers", () => {
  const first: ASv3Progress = {
    type: "asv3_progress",
    run_id: "r",
    event_id: "header-1",
    sequence: 1,
    language: "tr",
    phase: "tools",
    status: "running",
    title: "Ayniyet şartlarını inceliyorum",
    message:
      "İlgili genelgede yeşil hat için öngörülen istisnayı kontrol ediyorum.",
  };
  let state = applyASv3Progress(createASv3ProgressState(), first);
  state = applyASv3Progress(state, {
    ...first,
    event_id: "action-1",
    sequence: 2,
    task_id: "action:hashed-first",
    status: "completed",
  });
  state = applyASv3Progress(state, {
    ...first,
    event_id: "header-2",
    sequence: 3,
    title: "Teminat koşullarını inceliyorum",
    message:
      "Kanuni şartı ve teslimden sonra tamamlanacak işlemleri karşılaştırıyorum.",
  });
  state = applyASv3Progress(state, {
    ...state.header!,
    event_id: "action-2",
    sequence: 4,
    task_id: "action:hashed-second",
  });
  const { rerender } = render(
    <ASv3ProgressPanel state={state} stopped={false} />
  );
  fireEvent.click(
    screen.getByRole("button", { name: "Teminat koşullarını inceliyorum" })
  );
  expect(screen.queryByRole("tab")).not.toBeInTheDocument();
  expect(screen.getAllByTestId("asv3-past-step")).toHaveLength(2);
  expect(
    screen.getAllByRole("button", { name: "Ayniyet şartlarını inceliyorum" })
  ).toHaveLength(1);
  fireEvent.click(
    screen.getByRole("button", { name: "Ayniyet şartlarını inceliyorum" })
  );
  expect(screen.getByText(/genelgede yeşil hat/)).toBeInTheDocument();
  expect(screen.getByTestId("asv3-task-loading")).toBeInTheDocument();
  rerender(<ASv3ProgressPanel state={state} stopped />);
  expect(screen.queryByTestId("asv3-task-loading")).not.toBeInTheDocument();
});

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
  fireEvent.click(
    screen.getByRole("button", { name: "Garanti koşullarını inceliyorum" })
  );
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

it("starts collapsed during research, answer arrival and terminal replay without restarting activity", () => {
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
    screen.getByText("Araştırma tamamlandı").closest(".asv3-title-running")
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
  rerender(
    <ASv3ProgressPanel
      state={parallelState()}
      stopped={false}
      hasDisplayContent
    />
  );
  expect(screen.getAllByRole("tab")).toHaveLength(2);
  expect(screen.queryByTestId("asv3-task-loading")).not.toBeInTheDocument();
  expect(screen.getByRole("region", { name: "ASv3" })).toHaveAttribute(
    "aria-busy",
    "false"
  );
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
  fireEvent.click(
    screen.getByRole("button", { name: "Garanti koşullarını inceliyorum" })
  );
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

it("removes a pending placeholder when the answer arrives without a progress event", () => {
  const state = createASv3ProgressState();
  const { rerender } = render(
    <ASv3ProgressPanel state={state} stopped={false} pending />
  );
  expect(screen.getByRole("region", { name: "ASv3" })).toHaveTextContent(
    /^ASv3$/
  );
  rerender(
    <ASv3ProgressPanel
      state={state}
      stopped={false}
      pending
      hasDisplayContent
    />
  );
  expect(
    screen.queryByRole("region", { name: "ASv3" })
  ).not.toBeInTheDocument();
});

it("shows a title without an empty expandable body when no details were supplied", () => {
  const state = applyASv3Progress(createASv3ProgressState(), {
    type: "asv3_progress",
    run_id: "r",
    event_id: "title-only",
    sequence: 1,
    language: "tr",
    phase: "research",
    status: "running",
    title: "Kanunun geri geliş şartlarını inceliyorum",
    message: "   ",
  });
  render(<ASv3ProgressPanel state={state} stopped={false} />);
  expect(
    screen.getByText("Kanunun geri geliş şartlarını inceliyorum")
  ).toBeInTheDocument();
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
  expect(screen.queryByRole("tab")).not.toBeInTheDocument();
});

it("leaves no placeholder timeline above a direct answer without localized tool updates", () => {
  let state = applyASv3Progress(createASv3ProgressState(), {
    type: "asv3_progress",
    run_id: "r",
    event_id: "neutral-start",
    sequence: 1,
    language: "und",
    phase: "started",
    status: "running",
    title: "ASv3",
    message: "…",
  });
  const { rerender } = render(
    <ASv3ProgressPanel state={state} stopped={false} />
  );
  expect(screen.getByRole("region", { name: "ASv3" })).toHaveTextContent(
    /^ASv3$/
  );
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
  state = applyASv3Progress(state, {
    ...state.header!,
    event_id: "neutral-end",
    sequence: 2,
    phase: "completed",
    status: "completed",
  });
  rerender(
    <ASv3ProgressPanel state={state} stopped={false} hasDisplayContent />
  );
  expect(
    screen.queryByRole("region", { name: "ASv3" })
  ).not.toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: title }));
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
    screen.queryByText(/Gümrük Kanunu'ndaki bedelsiz tamir şartlarını/)
  ).not.toBeInTheDocument();
  fireEvent.click(
    screen.getByRole("button", {
      name: "Garanti kapsamındaki iki ihtimali inceliyorum",
    })
  );
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
  expect(
    screen.queryByRole("button", { name: "Araştırmaya devam et" })
  ).not.toBeInTheDocument();
});

it("opens the current description and real past steps with keyboard even with zero researchers", async () => {
  const user = setupUser();
  let state = applyASv3Progress(createASv3ProgressState(), {
    type: "asv3_progress",
    run_id: "r",
    event_id: "initial",
    sequence: 1,
    language: "tr",
    phase: "started",
    status: "running",
    title: "Geri gelen makinelerin koşullarını inceliyorum",
    message:
      "Ayniyet ve ihracatta alınan iadenin durumunu ayrı ayrı kontrol ediyorum.",
  });
  state = applyASv3Progress(state, {
    ...state.header!,
    event_id: "reading",
    sequence: 2,
    phase: "research",
    title: "Geri geliş belgelerini inceliyorum",
    message:
      "Yeşil hat kolaylığının teslim ve teminat koşullarını nasıl etkilediğini kontrol ediyorum.",
  });
  render(<ASv3ProgressPanel state={state} stopped={false} />);
  const toggle = screen.getByRole("button", {
    name: "Geri geliş belgelerini inceliyorum",
  });
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText(/Yeşil hat kolaylığının/)).not.toBeInTheDocument();
  expect(screen.queryByRole("tab")).not.toBeInTheDocument();
  await user.tab();
  expect(toggle).toHaveFocus();
  await user.keyboard("{Enter}");
  expect(toggle).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByText(/Yeşil hat kolaylığının/)).toBeInTheDocument();
  const earlier = screen.getByRole("button", {
    name: "Geri gelen makinelerin koşullarını inceliyorum",
  });
  expect(screen.queryByText(/Ayniyet ve ihracatta/)).not.toBeInTheDocument();
  await user.tab();
  expect(earlier).toHaveFocus();
  await user.keyboard(" ");
  expect(earlier).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByText(/Ayniyet ve ihracatta/)).toBeInTheDocument();
  await user.click(toggle);
  expect(screen.queryByText(/Ayniyet ve ihracatta/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Yeşil hat kolaylığının/)).not.toBeInTheDocument();
  expect(screen.queryByRole("tab")).not.toBeInTheDocument();
});

it("retains actual step history on terminal replay while leaving all animation inactive", () => {
  const initial: ASv3Progress = {
    type: "asv3_progress",
    run_id: "r",
    event_id: "initial",
    sequence: 1,
    language: "en",
    phase: "research",
    status: "running",
    title: "Checking the guarantee conditions",
    message: "Reviewing the original repair provision.",
  };
  let state = applyASv3Progress(createASv3ProgressState(), initial);
  state = applyASv3Progress(state, {
    ...initial,
    event_id: "final",
    sequence: 2,
    status: "completed",
    title: "The review is complete",
    message: "The answer is ready.",
  });
  render(<ASv3ProgressPanel state={state} stopped />);
  const toggle = screen.getByRole("button", { name: "The review is complete" });
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  expect(toggle.closest(".asv3-title-running")).toBeNull();
  expect(screen.queryByText("The answer is ready.")).not.toBeInTheDocument();
  fireEvent.click(toggle);
  expect(screen.getByText("The answer is ready.")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Checking the guarantee conditions" })
  ).toBeInTheDocument();
  expect(screen.queryByTestId("asv3-task-loading")).not.toBeInTheDocument();
});
