import { render, screen } from "@tests/setup/test-utils";
import useSWR from "swr";
import AmendmentRuntimePanel from "@/sections/cards/AmendmentRuntimePanel";
import type { AmendmentRuntime } from "@/lib/regulatory/interfaces";

jest.mock("swr", () => ({
  ...jest.requireActual("swr"),
  __esModule: true,
  default: jest.fn(),
}));

const sample: AmendmentRuntime = {
  batch_id: 1,
  status: "analyzing",
  stage: "processing",
  lease_generation: 4,
  scope: "worker_container",
  raw_text_chars: 5633,
  current_bytes: 2.5 * 1024 ** 3,
  limit_bytes: 5 * 1024 ** 3,
  peak_bytes: 3 * 1024 ** 3,
  reserve_bytes: 500 * 1024 ** 2,
  active: 4,
  peak_active: 10,
  max_parallel: 10,
  admission_limited: false,
  dependency_limited: false,
  calibrating: false,
  memory_checked_at: null,
  activity_checked_at: null,
};

test("current memory does not make an old activity count appear live", () => {
  jest.mocked(useSWR).mockReturnValue({
    data: {
      ...sample,
      memory_checked_at: new Date().toISOString(),
      activity_checked_at: new Date(Date.now() - 60000).toISOString(),
    },
    error: undefined,
    isLoading: false,
    isValidating: false,
    mutate: jest.fn(),
  });
  render(<AmendmentRuntimePanel batchId={1} status="analyzing" />);
  expect(
    screen.getByText(/Worker memory: 2.50 GiB \/ 5.00 GiB/)
  ).toBeInTheDocument();
  expect(screen.getByText(/Last task count: 4/)).toBeInTheDocument();
  expect(screen.queryByText(/Active tasks:/)).not.toBeInTheDocument();
  expect(screen.getByText(/Task count is not current/)).toBeInTheDocument();
});

test("completed batches show last measurements and stop polling", () => {
  jest.mocked(useSWR).mockReturnValue({
    data: {
      ...sample,
      status: "analyzed",
      memory_checked_at: new Date().toISOString(),
      activity_checked_at: new Date().toISOString(),
    },
    error: undefined,
    isLoading: false,
    isValidating: false,
    mutate: jest.fn(),
  });
  render(<AmendmentRuntimePanel batchId={1} status="analyzed" />);
  expect(screen.getByText(/Last worker memory:/)).toBeInTheDocument();
  const options = jest.mocked(useSWR).mock.calls.at(-1)?.[2];
  const refresh = options?.refreshInterval;
  if (typeof refresh !== "function")
    throw new Error("Expected conditional polling");
  expect(refresh(sample)).toBe(0);
});

test("unavailable telemetry never shows zero consumption", () => {
  jest.mocked(useSWR).mockReturnValue({
    data: undefined,
    error: new Error("unavailable"),
    isLoading: false,
    isValidating: false,
    mutate: jest.fn(),
  });
  render(<AmendmentRuntimePanel batchId={2} status="analyzing" />);
  expect(screen.getByText(/measurements are unavailable/)).toBeInTheDocument();
  expect(screen.queryByText(/0.00 GiB/)).not.toBeInTheDocument();
});

test("dependency waiting is not shown as a memory shortage", () => {
  jest.mocked(useSWR).mockReturnValue({
    data: {
      ...sample,
      dependency_limited: true,
      activity_checked_at: new Date().toISOString(),
    },
    error: undefined,
    isLoading: false,
    isValidating: false,
    mutate: jest.fn(),
  });
  render(<AmendmentRuntimePanel batchId={1} status="analyzing" />);
  expect(screen.getByText(/same provision are waiting/)).toBeInTheDocument();
  expect(
    screen.queryByText(/Waiting for memory capacity/)
  ).not.toBeInTheDocument();
});
