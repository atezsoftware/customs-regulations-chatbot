import {apiFetch} from './api';
import type {
  AuthTokens,
  ChatMessageRecord,
  ChatSession,
  Directory,
  DirectoryDetail,
  DirectoryIndexStatus,
  DirectoryFile,
  FileChunksResponse,
  SafeUser,
  SessionFile,
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
  linkedDirectories: (id: number) =>
    apiFetch<Directory[]>(`/chat-sessions/${id}/directories`),
  setLinkedDirectories: (id: number, directoryIds: number[]) =>
    apiFetch<void>(`/chat-sessions/${id}/directories`, {
      method: 'PUT',
      body: {directoryIds},
    }),
  visibleFiles: (id: number) => apiFetch<SessionFile[]>(`/chat-sessions/${id}/files`),
  messages: (id: number) => apiFetch<ChatMessageRecord[]>(`/chat-sessions/${id}/messages`),
  sendMessage: (id: number, input: {content: string; model?: string; temperature?: number}) =>
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
