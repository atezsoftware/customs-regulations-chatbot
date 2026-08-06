import type { FederatedConnectorConfig } from "@/lib/types";
import type { CategorizedFiles, ProjectFile } from "@/lib/projects/types";

const documentSetFilesEndpoint = (documentSetId: number) =>
  `/api/manage/admin/document-set/${documentSetId}/files`;

const documentSetFileEndpoint = (documentSetId: number, userFileId: string) =>
  `${documentSetFilesEndpoint(documentSetId)}/${encodeURIComponent(userFileId)}`;

function throwDocumentSetFileRequestError(
  action: string,
  response: Response
): never {
  throw new Error(`${action} failed (Status: ${response.status})`);
}

export interface DocumentSetCreationRequest {
  name: string;
  description: string;
  cc_pair_ids: number[];
  is_public: boolean;
  users: string[];
  groups: number[];
  federated_connectors: FederatedConnectorConfig[];
}

export const createDocumentSet = async ({
  name,
  description,
  cc_pair_ids,
  is_public,
  users,
  groups,
  federated_connectors,
}: DocumentSetCreationRequest) => {
  return fetch("/api/manage/admin/document-set", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      name,
      description,
      cc_pair_ids,
      is_public,
      users,
      groups,
      federated_connectors,
    }),
  });
};

interface DocumentSetUpdateRequest {
  id: number;
  name: string;
  description: string;
  cc_pair_ids: number[];
  is_public: boolean;
  users: string[];
  groups: number[];
  federated_connectors: FederatedConnectorConfig[];
}

export const updateDocumentSet = async ({
  id,
  name,
  description,
  cc_pair_ids,
  is_public,
  users,
  groups,
  federated_connectors,
}: DocumentSetUpdateRequest) => {
  return fetch("/api/manage/admin/document-set", {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      id,
      name,
      description,
      cc_pair_ids,
      is_public,
      users,
      groups,
      federated_connectors,
    }),
  });
};

export const deleteDocumentSet = async (id: number) => {
  return fetch(`/api/manage/admin/document-set/${id}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
  });
};

export function getDocumentSetFilesKey(documentSetId: number): string {
  return documentSetFilesEndpoint(documentSetId);
}

export async function getDocumentSetFiles(
  documentSetId: number
): Promise<ProjectFile[]> {
  const response = await fetch(documentSetFilesEndpoint(documentSetId));
  if (!response.ok) {
    throwDocumentSetFileRequestError("Fetch document set files", response);
  }
  return response.json();
}

export async function uploadDocumentSetFiles(
  documentSetId: number,
  files: File[],
  tempIdMap?: Map<string, string>
): Promise<CategorizedFiles> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  if (tempIdMap) {
    formData.append(
      "temp_id_map",
      JSON.stringify(Object.fromEntries(tempIdMap))
    );
  }

  const response = await fetch(
    `/api/manage/admin/document-set/${documentSetId}/file/upload`,
    {
      method: "POST",
      body: formData,
    }
  );
  if (!response.ok) {
    throwDocumentSetFileRequestError("Upload document set files", response);
  }
  return response.json();
}

export async function linkFileToDocumentSet(
  documentSetId: number,
  userFileId: string
): Promise<void> {
  const response = await fetch(
    documentSetFileEndpoint(documentSetId, userFileId),
    { method: "POST" }
  );
  if (!response.ok) {
    throwDocumentSetFileRequestError("Link file to document set", response);
  }
}

export async function unlinkFileFromDocumentSet(
  documentSetId: number,
  userFileId: string
): Promise<void> {
  const response = await fetch(
    documentSetFileEndpoint(documentSetId, userFileId),
    { method: "DELETE" }
  );
  if (!response.ok) {
    throwDocumentSetFileRequestError("Remove file from document set", response);
  }
}
