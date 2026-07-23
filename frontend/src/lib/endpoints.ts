import {apiFetch} from './api';
import type {
  AdminSupportSessionDetail,
  AdminSupportSessionsResponse,
  AmendmentBatch,
  AmendmentProposal,
  AnalyzeAmendmentResult,
  ApproveAmendmentResult,
  AuthTokens,
  BenchmarkQuestion,
  BenchmarkRun,
  BenchmarkRunDetail,
  BenchmarkRunItem,
  ChatMessageRecord,
  ChatSession,
  Directory,
  DirectoryDetail,
  DirectoryIndexStatus,
  DirectoryFile,
  FileChunksResponse,
  SafeUser,
  SessionFile,
  UsageAnalytics,
  UsageRange,
  LlmModelsResponse,
} from '../types';

export const authApi = {
  register: (input: {email: string; password: string; fullName?: string}) =>
    apiFetch<{user: SafeUser; tokens: AuthTokens}>('/auth/register', {
      method: 'POST',
      body: input,
    }),
  login: (input: {email: string; password: string}) =>
    apiFetch<{user: SafeUser; tokens: AuthTokens}>('/auth/login', {
      method: 'POST',
      body: input,
    }),
  me: () => apiFetch<SafeUser>('/auth/me'),
  logout: (refreshToken: string) =>
    apiFetch<void>('/auth/logout', {method: 'POST', body: {refreshToken}}),
  changePassword: (input: {currentPassword: string; newPassword: string}) =>
    apiFetch<void>('/auth/change-password', {method: 'POST', body: input}),
};

export const directoriesApi = {
  list: () => apiFetch<Directory[]>('/directories'),
  get: (id: number) => apiFetch<DirectoryDetail>(`/directories/${id}`),
  create: (name: string) =>
    apiFetch<Directory>('/directories', {method: 'POST', body: {name}}),
  rename: (id: number, name: string) =>
    apiFetch<void>(`/directories/${id}`, {method: 'PATCH', body: {name}}),
  remove: (id: number) => apiFetch<void>(`/directories/${id}`, {method: 'DELETE'}),
  upload: (id: number, files: File[]) => {
    const form = new FormData();
    files.forEach(file => form.append('files', file));
    return apiFetch<DirectoryFile[]>(`/directories/${id}/files`, {
      method: 'POST',
      body: form,
      isForm: true,
    });
  },
  renameFile: (directoryId: number, fileId: number, name: string) =>
    apiFetch<void>(`/directories/${directoryId}/files/${fileId}`, {
      method: 'PATCH',
      body: {name},
    }),
  removeFile: (directoryId: number, fileId: number) =>
    apiFetch<void>(`/directories/${directoryId}/files/${fileId}`, {method: 'DELETE'}),
  indexStatus: (id: number) =>
    apiFetch<DirectoryIndexStatus>(`/directories/${id}/index/status`),
  startChunking: (id: number) =>
    apiFetch<DirectoryIndexStatus>(`/directories/${id}/chunks`, {method: 'POST'}),
  startIndex: (id: number) =>
    apiFetch<DirectoryIndexStatus>(`/directories/${id}/index`, {method: 'POST'}),
  fileChunks: (directoryId: number, fileId: number) =>
    apiFetch<FileChunksResponse>(`/directories/${directoryId}/files/${fileId}/chunks`),
};

export const chatSessionsApi = {
  list: () => apiFetch<ChatSession[]>('/chat-sessions'),
  create: (title?: string) =>
    apiFetch<ChatSession>('/chat-sessions', {method: 'POST', body: {title}}),
  rename: (id: number, title: string) =>
    apiFetch<void>(`/chat-sessions/${id}`, {method: 'PATCH', body: {title}}),
  remove: (id: number) => apiFetch<void>(`/chat-sessions/${id}`, {method: 'DELETE'}),
  setModel: (id: number, input: {provider: string; modelId: string}) =>
    apiFetch<ChatSession>(`/chat-sessions/${id}/model`, {method: 'PATCH', body: input}),
  linkedDirectories: (id: number) =>
    apiFetch<Directory[]>(`/chat-sessions/${id}/directories`),
  setLinkedDirectories: (id: number, directoryIds: number[]) =>
    apiFetch<void>(`/chat-sessions/${id}/directories`, {
      method: 'PUT',
      body: {directoryIds},
    }),
  visibleFiles: (id: number) => apiFetch<SessionFile[]>(`/chat-sessions/${id}/files`),
  messages: (id: number) => apiFetch<ChatMessageRecord[]>(`/chat-sessions/${id}/messages`),
  sendMessage: (id: number, input: {content: string; temperature?: number}) =>
    apiFetch<{
      messageId: number;
      userMessage: ChatMessageRecord;
      assistantMessage: ChatMessageRecord;
    }>(`/chat-sessions/${id}/messages`, {
      method: 'POST',
      body: input,
    }),
  cancelMessage: (sessionId: number, messageId: number) =>
    apiFetch<void>(`/chat-sessions/${sessionId}/messages/${messageId}/cancel`, {
      method: 'POST',
    }),
};

export const llmModelsApi = {
  list: () => apiFetch<LlmModelsResponse>('/llm/models'),
};

export const usageApi = {
  get: (range: UsageRange = '30d') =>
    apiFetch<UsageAnalytics>(`/analytics/usage?range=${encodeURIComponent(range)}`),
};

export const adminSupportApi = {
  sessions: (input: {search?: string; limit?: number} = {}) => {
    const params = new URLSearchParams();
    if (input.search?.trim()) params.set('search', input.search.trim());
    if (input.limit) params.set('limit', String(input.limit));
    const query = params.toString();
    return apiFetch<AdminSupportSessionsResponse>(
      `/admin/support/sessions${query ? `?${query}` : ''}`,
    );
  },
  messages: (sessionId: number) =>
    apiFetch<AdminSupportSessionDetail>(`/admin/support/sessions/${sessionId}/messages`),
};

export const benchmarkApi = {
  questions: {
    list: () => apiFetch<{questions: BenchmarkQuestion[]}>('/admin/benchmark/questions'),
    create: (
      input: Partial<Omit<BenchmarkQuestion, 'id'>> & {prompt: string},
    ) =>
      apiFetch<BenchmarkQuestion>('/admin/benchmark/questions', {
        method: 'POST',
        body: input,
      }),
    update: (id: number, input: Partial<Omit<BenchmarkQuestion, 'id'>>) =>
      apiFetch<BenchmarkQuestion>(`/admin/benchmark/questions/${id}`, {
        method: 'PATCH',
        body: input,
      }),
    remove: (id: number) =>
      apiFetch<void>(`/admin/benchmark/questions/${id}`, {method: 'DELETE'}),
  },
  runs: {
    list: () => apiFetch<{runs: BenchmarkRun[]}>('/admin/benchmark/runs'),
    create: (input: {
      label?: string;
      providerModelPairs: Array<{provider: string; modelId: string}>;
      questionIds: number[] | 'all-active';
      judgeProvider: string;
      judgeModel: string;
    }) =>
      apiFetch<{runId: number}>('/admin/benchmark/runs', {
        method: 'POST',
        body: input,
      }),
    get: (id: number) => apiFetch<BenchmarkRunDetail>(`/admin/benchmark/runs/${id}`),
    items: (id: number) =>
      apiFetch<{items: BenchmarkRunItem[]}>(`/admin/benchmark/runs/${id}/items`),
    cancel: (id: number) =>
      apiFetch<void>(`/admin/benchmark/runs/${id}/cancel`, {method: 'POST'}),
  },
};

export const amendmentsApi = {
  analyze: (directoryId: number, rawText: string) =>
    apiFetch<AnalyzeAmendmentResult>('/admin/amendments/analyze', {
      method: 'POST',
      body: {directoryId, rawText},
    }),
  listProposals: (input: {status?: string; directoryId?: number} = {}) => {
    const params = new URLSearchParams();
    if (input.status) params.set('status', input.status);
    if (input.directoryId !== undefined) params.set('directoryId', String(input.directoryId));
    const query = params.toString();
    return apiFetch<{proposals: AmendmentProposal[]}>(
      `/admin/amendments/proposals${query ? `?${query}` : ''}`,
    );
  },
  getBatch: (batchId: string) =>
    apiFetch<{batch: AmendmentBatch; proposals: AmendmentProposal[]}>(
      `/admin/amendments/batches/${batchId}`,
    ),
  approve: (proposalIds: string[]) =>
    apiFetch<ApproveAmendmentResult>('/admin/amendments/approve', {
      method: 'POST',
      body: {proposalIds},
    }),
  reject: (proposalId: string) =>
    apiFetch<{proposal: AmendmentProposal}>(`/admin/amendments/proposals/${proposalId}/reject`, {
      method: 'POST',
    }),
  remove: (proposalId: string) =>
    apiFetch<{deleted: boolean}>(`/admin/amendments/proposals/${proposalId}`, {
      method: 'DELETE',
    }),
};
