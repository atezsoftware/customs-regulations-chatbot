"use client";

import { useEffect, useRef, useState } from "react";
import { Button, Text } from "@opal/components";
import type { AmendmentSourcePackage } from "@/lib/regulatory/amendments";
import {
  type MarkdownDraft,
  type MarkdownDraftDocument,
  MarkdownDraftRequestError,
  createMarkdownDraft,
  getMarkdownDraft,
  listMarkdownDrafts,
  prepareMarkdownDraft,
  removeMarkdownDraftDocument,
  uploadMarkdownDraftDocuments,
} from "@/lib/regulatory/markdownDrafts";

interface MarkdownDraftStagingProps {
  disabled: boolean;
  preparationEnabled: boolean;
  onPrepared: (source: AmendmentSourcePackage) => Promise<void>;
}

interface PendingUpload {
  files: File[];
  parentId: string | null;
  key: string;
  draft: MarkdownDraft | null;
}

const creationKeyStorage = "regulatory-markdown-draft-creation-key";

function savedCreationKey(): string {
  try {
    const existing = localStorage.getItem(creationKeyStorage);
    if (existing) return existing;
  } catch {
    // Storage may be unavailable in a private browser session.
  }
  const key = globalThis.crypto.randomUUID();
  try {
    localStorage.setItem(creationKeyStorage, key);
  } catch {
    // The current mounted form still retains its retry key.
  }
  return key;
}

function failureMessage(error: unknown): string {
  return error instanceof MarkdownDraftRequestError
    ? error.message
    : "İşlem tamamlanamadı. Bağlantınızı kontrol edip yeniden deneyin.";
}

export default function MarkdownDraftStaging({
  disabled,
  preparationEnabled,
  onPrepared,
}: MarkdownDraftStagingProps) {
  const [drafts, setDrafts] = useState<MarkdownDraft[]>([]);
  const [draft, setDraft] = useState<MarkdownDraft | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(
    null
  );
  const [prepareAttempted, setPrepareAttempted] = useState(false);
  const [parentId, setParentId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const annexInputRef = useRef<HTMLInputElement>(null);
  const creationKey = useRef<string | null>(null);
  const preparedSource = useRef<AmendmentSourcePackage | null>(null);
  const mounted = useRef(true);
  const operationLock = useRef(false);
  const requestEpoch = useRef(0);

  useEffect(() => {
    mounted.current = true;
    void loadDrafts();
    return () => {
      mounted.current = false;
      requestEpoch.current += 1;
    };
    // Draft loading runs once per mounted Markdown form.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function isCurrent(epoch: number): boolean {
    return mounted.current && requestEpoch.current === epoch;
  }

  function remember(snapshot: MarkdownDraft, epoch: number) {
    if (!isCurrent(epoch)) return;
    setDraft(snapshot);
    setDrafts((current) => [
      snapshot,
      ...current.filter((item) => item.id !== snapshot.id),
    ]);
  }

  async function loadDrafts() {
    const epoch = ++requestEpoch.current;
    setBusy(true);
    setError(null);
    try {
      const snapshots = await listMarkdownDrafts();
      if (!isCurrent(epoch)) return;
      const selected = draft
        ? (snapshots.find((item) => item.id === draft.id) ?? draft)
        : (snapshots.find((item) => item.status === "draft") ??
          snapshots[0] ??
          null);
      if (selected?.id !== draft?.id) preparedSource.current = null;
      setDrafts(snapshots);
      setDraft(selected);
      setPrepareAttempted(false);
    } catch (error) {
      if (isCurrent(epoch)) setError(failureMessage(error));
    } finally {
      if (isCurrent(epoch)) setBusy(false);
    }
  }

  async function upload(operation: PendingUpload) {
    if (operationLock.current || disabled) return;
    operationLock.current = true;
    const epoch = ++requestEpoch.current;
    setBusy(true);
    setError(null);
    setPendingUpload(operation);
    try {
      let destination = operation.draft;
      if (!destination) {
        creationKey.current ??= savedCreationKey();
        destination = await createMarkdownDraft(creationKey.current);
        if (!isCurrent(epoch)) return;
        operation = { ...operation, draft: destination };
        if (isCurrent(epoch)) setPendingUpload(operation);
        remember(destination, epoch);
        try {
          localStorage.removeItem(creationKeyStorage);
        } catch {
          // Draft recovery also works through the server's recent list.
        }
        creationKey.current = null;
      }
      const uploaded = await uploadMarkdownDraftDocuments(
        destination,
        operation.files,
        operation.key,
        operation.parentId
      );
      remember(uploaded, epoch);
      if (isCurrent(epoch)) setPendingUpload(null);
    } catch (error) {
      if (isCurrent(epoch)) {
        setError(failureMessage(error));
        if (
          error instanceof MarkdownDraftRequestError &&
          [400, 401, 403, 404, 409, 413, 415, 422].includes(error.status)
        ) {
          setPendingUpload(null);
        }
      }
    } finally {
      operationLock.current = false;
      if (isCurrent(epoch)) setBusy(false);
    }
  }

  function chooseFiles(files: FileList | null, parent: string | null) {
    if (!files?.length) return;
    const selected = Array.from(files);
    if (selected.some((file) => !file.name.toLowerCase().endsWith(".md"))) {
      setError("Lütfen yalnızca .md uzantılı Markdown dosyaları seçin.");
      return;
    }
    void upload({
      files: selected,
      parentId: parent,
      key: globalThis.crypto.randomUUID(),
      draft,
    });
  }

  async function remove(document: MarkdownDraftDocument) {
    if (!draft || operationLock.current || frozen || disabled) return;
    operationLock.current = true;
    const epoch = ++requestEpoch.current;
    setBusy(true);
    setError(null);
    try {
      remember(await removeMarkdownDraftDocument(draft, document.id), epoch);
    } catch (error) {
      // A completed deletion can lose its response; reload the authoritative tree.
      try {
        const snapshot = await getMarkdownDraft(draft.id);
        remember(snapshot, epoch);
        if (
          snapshot.documents.some((item) => item.id === document.id) &&
          isCurrent(epoch)
        )
          setError(failureMessage(error));
      } catch {
        if (isCurrent(epoch)) setError(failureMessage(error));
      }
    } finally {
      operationLock.current = false;
      if (isCurrent(epoch)) setBusy(false);
    }
  }

  async function prepare() {
    if (!draft || !preparationEnabled || operationLock.current || disabled)
      return;
    operationLock.current = true;
    const epoch = ++requestEpoch.current;
    setBusy(true);
    setError(null);
    setPrepareAttempted(true);
    try {
      const source =
        preparedSource.current ?? (await prepareMarkdownDraft(draft));
      if (!isCurrent(epoch)) return;
      preparedSource.current = source;
      remember(
        { ...draft, status: "prepared", source_package_id: source.id },
        epoch
      );
      await onPrepared(source);
    } catch (error) {
      if (!isCurrent(epoch)) return;
      if (!preparedSource.current) {
        try {
          const snapshot = await getMarkdownDraft(draft.id);
          if (!isCurrent(epoch)) return;
          remember(snapshot, epoch);
          setPrepareAttempted(snapshot.status === "prepared");
        } catch {
          // An unknown preparation result stays frozen until an authoritative reload.
        }
      }
      if (isCurrent(epoch)) setError(failureMessage(error));
    } finally {
      operationLock.current = false;
      if (isCurrent(epoch)) setBusy(false);
    }
  }

  const frozen = draft?.status === "prepared" || prepareAttempted;
  const mutationDisabled = disabled || busy || frozen || Boolean(pendingUpload);
  const parent = draft?.documents.find((document) => document.id === parentId);

  function renderDocument(document: MarkdownDraftDocument) {
    return (
      <section
        key={document.id}
        aria-label={document.display_name}
        className="flex flex-col gap-2 rounded-12 border border-border-02 p-3"
      >
        <Text font="main-ui-action">{document.display_name}</Text>
        <div className="flex flex-wrap gap-2">
          <Button
            prominence="secondary"
            disabled={mutationDisabled}
            onClick={() => {
              setParentId(document.id);
              annexInputRef.current?.click();
            }}
          >
            Belgeye ek ekle
          </Button>
          <Button
            prominence="tertiary"
            disabled={mutationDisabled}
            onClick={() => void remove(document)}
          >
            Belgeyi kaldır
          </Button>
        </div>
        {draft?.documents
          .filter((item) => item.parent_document_id === document.id)
          .map(renderDocument)}
      </section>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <Text as="p" font="secondary-body">
        Önce Markdown belgelerinizi ekleyin. Her belgenin eklerini “Belgeye ek
        ekle” ile seçin. Tüm dosyalar hazır olduğunda chunkları hazırlayıp
        inceleyebilirsiniz.
      </Text>
      <Text as="p" font="secondary-body">
        Yüklenen dosyalar taslak olarak saklanır. Külliyata eklemek için
        hazırlama ve incelemenin ardından onay vermeniz gerekir.
      </Text>
      <Text as="p" font="secondary-body">
        Ekler dahil en fazla 21 belge ekleyebilirsiniz. Her dosya en fazla 2
        MiB, taslağın toplam boyutu en fazla 32 MiB olabilir.
      </Text>
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        multiple
        accept=".md"
        aria-label="Markdown belgeleri"
        disabled={mutationDisabled}
        onChange={(event) => {
          chooseFiles(event.target.files, null);
          event.target.value = "";
        }}
      />
      <input
        ref={annexInputRef}
        type="file"
        className="hidden"
        multiple
        accept=".md"
        aria-label={
          parent
            ? `${parent.display_name} için ek belgeler`
            : "Ek Markdown belgeleri"
        }
        disabled={mutationDisabled}
        onChange={(event) => {
          chooseFiles(event.target.files, parentId);
          event.target.value = "";
        }}
      />
      <div className="flex flex-wrap gap-2">
        <Button
          prominence="secondary"
          disabled={mutationDisabled}
          onClick={() => inputRef.current?.click()}
        >
          Markdown belgelerini ekle
        </Button>
        <Button
          prominence="tertiary"
          disabled={disabled || busy || Boolean(pendingUpload)}
          onClick={() => void loadDrafts()}
        >
          Taslakları yenile
        </Button>
        {draft && (
          <Button
            prominence="tertiary"
            disabled={disabled || busy || Boolean(pendingUpload)}
            onClick={() => {
              requestEpoch.current += 1;
              setDraft(null);
              setPrepareAttempted(false);
              preparedSource.current = null;
              creationKey.current = null;
              try {
                localStorage.removeItem(creationKeyStorage);
              } catch {
                /* Storage is optional. */
              }
              setError(null);
            }}
          >
            Yeni taslak
          </Button>
        )}
      </div>
      {drafts.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {drafts.map((item) => (
            <Button
              key={item.id}
              prominence={draft?.id === item.id ? "primary" : "secondary"}
              disabled={disabled || busy || Boolean(pendingUpload)}
              onClick={() => {
                requestEpoch.current += 1;
                setDraft(item);
                setPrepareAttempted(false);
                preparedSource.current = null;
                setError(null);
              }}
            >{`${item.documents[0]?.display_name ?? "Boş taslak"} · ${new Date(item.updated_at).toLocaleString("tr-TR")}`}</Button>
          ))}
        </div>
      )}
      {draft?.documents
        .filter((document) => !document.parent_document_id)
        .map(renderDocument)}
      {frozen && (
        <Text as="p" font="secondary-body">
          {draft?.status === "prepared"
            ? "Belgeler hazırlama için kaydedildi. Bu taslağın dosyaları artık değiştirilemez."
            : busy
              ? "Hazırlama tamamlanana kadar dosyalar değiştirilemez."
              : "Hazırlama sonucu henüz doğrulanamadı. Yeniden deneyin veya taslağı yenileyin."}
        </Text>
      )}
      {busy && (
        <div role="status">
          <Text font="secondary-body">İşlem sürüyor…</Text>
        </div>
      )}
      {error && <Text color="status-error-05">{error}</Text>}
      {pendingUpload && !busy && (
        <div>
          <Button
            disabled={disabled}
            onClick={() => void upload(pendingUpload)}
          >
            Yüklemeyi tekrar dene
          </Button>
        </div>
      )}
      <div>
        <Button
          disabled={
            disabled ||
            busy ||
            !draft?.documents.length ||
            !preparationEnabled ||
            Boolean(pendingUpload)
          }
          onClick={() => void prepare()}
        >
          {frozen ? "Hazırlamayı tekrar dene" : "Chunkları hazırla"}
        </Button>
      </div>
    </div>
  );
}
