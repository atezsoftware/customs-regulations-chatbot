"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  Button,
  InputSelect,
  InputTextArea,
  InputTypeIn,
  Tag,
  Text,
} from "@opal/components";
import { SettingsLayouts, toast } from "@opal/layouts";
import SvgHistory from "@opal/icons/history";
import { useDocumentSets } from "@/lib/hooks/useDocumentSets";
import {
  AmendmentBatch,
  AmendmentProposal,
  analyzeAmendment,
  approveProposal,
  extractAmendmentPdf,
  extractAmendmentUrl,
  listAmendmentBatches,
  listAmendmentProposals,
  rejectProposal,
} from "@/lib/regulatory/amendments";

type AmendmentSourceMode = "text" | "url" | "pdf";

function sourceIdentity(
  mode: AmendmentSourceMode,
  url: string,
  file: File | null
) {
  if (mode === "url") return url.trim() ? `url:${url.trim()}` : null;
  if (mode === "pdf" && file) {
    return `pdf:${file.name}:${file.size}:${file.lastModified}`;
  }
  return null;
}

function ChunkPreview({
  title,
  chunk,
  validityStart,
  validityEnd,
}: {
  title: string;
  chunk: Record<string, unknown> | null;
  validityStart?: string | null;
  validityEnd?: string | null;
}) {
  const text = chunk ? String(chunk.text ?? "") : null;
  const headingPath = chunk?.heading_path;
  const headingPathText = Array.isArray(headingPath)
    ? headingPath.join(" > ")
    : "";

  return (
    <div className="flex-1 rounded-lg border border-border-02 p-3 flex flex-col gap-2 min-w-0">
      <Text font="main-ui-action" color="text-04">
        {title}
      </Text>
      {chunk === null ? (
        <Text font="main-ui-body" color="text-03">
          (new — no existing chunk)
        </Text>
      ) : (
        <>
          {headingPathText && (
            <Text font="secondary-body" color="text-03">
              {headingPathText}
            </Text>
          )}
          <Text font="main-ui-body" color="text-05" as="p">
            {text ?? ""}
          </Text>
          {(validityStart || validityEnd) && (
            <Text font="secondary-body" color="text-03">
              {`Validity: ${validityStart ?? "—"} → ${validityEnd ?? "open"}`}
            </Text>
          )}
        </>
      )}
    </div>
  );
}

function ProposalCard({
  proposal,
  onDecided,
}: {
  proposal: AmendmentProposal;
  onDecided: () => void;
}) {
  const [deciding, setDeciding] = useState(false);

  const handleApprove = useCallback(async () => {
    setDeciding(true);
    try {
      await approveProposal(proposal.id);
      toast.success("Proposal approved and indexed.");
      onDecided();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Approve failed.");
    } finally {
      setDeciding(false);
    }
  }, [proposal.id, onDecided]);

  const handleReject = useCallback(async () => {
    setDeciding(true);
    try {
      await rejectProposal(proposal.id);
      toast.success("Proposal rejected.");
      onDecided();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Reject failed.");
    } finally {
      setDeciding(false);
    }
  }, [proposal.id, onDecided]);

  const isNewChunk = Object.keys(proposal.old_chunk_snapshot).length === 0;
  const newDraft = proposal.new_chunk_draft;

  return (
    <div className="rounded-lg border border-border-02 p-4 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-2">
        <Text font="main-ui-body" color="text-05" as="p">
          {proposal.instruction_text}
        </Text>
        <div className="flex items-center gap-2 shrink-0">
          {isNewChunk && <Tag title="New article" />}
          {proposal.duplicate_target && <Tag title="Duplicate target" />}
          <Tag title={proposal.status} />
        </div>
      </div>

      {proposal.match_rationale && (
        <Text font="secondary-body" color="text-03">
          {`Match: ${proposal.match_rationale}${
            proposal.match_confidence != null
              ? ` (confidence ${(proposal.match_confidence * 100).toFixed(0)}%)`
              : ""
          }`}
        </Text>
      )}

      <div className="flex gap-3">
        <ChunkPreview
          title="Current"
          chunk={isNewChunk ? null : proposal.old_chunk_snapshot}
        />
        <ChunkPreview
          title="Proposed"
          chunk={newDraft}
          validityStart={newDraft.effective_start_date as string | undefined}
          validityEnd={newDraft.effective_end_date as string | undefined}
        />
      </div>

      {proposal.date_rationale && (
        <Text font="secondary-body" color="text-03">
          {`Dates: ${proposal.date_rationale}`}
        </Text>
      )}

      {proposal.status === "pending" && (
        <div className="flex gap-2 justify-end">
          <Button
            variant="danger"
            prominence="secondary"
            onClick={() => void handleReject()}
            disabled={deciding}
          >
            Reject
          </Button>
          <Button onClick={() => void handleApprove()} disabled={deciding}>
            Approve
          </Button>
        </div>
      )}
    </div>
  );
}

export default function AmendmentsPage() {
  const { documentSets } = useDocumentSets();
  const [selectedDocumentSetId, setSelectedDocumentSetId] = useState<
    string | null
  >(null);
  const [rawText, setRawText] = useState("");
  const [sourceMode, setSourceMode] = useState<AmendmentSourceMode>("text");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [extractedSourceIdentity, setExtractedSourceIdentity] = useState<
    string | null
  >(null);
  const [extracting, setExtracting] = useState(false);
  const pdfInputRef = useRef<HTMLInputElement>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const [batches, setBatches] = useState<AmendmentBatch[]>([]);
  const [selectedBatchId, setSelectedBatchId] = useState<number | null>(null);
  const [proposals, setProposals] = useState<AmendmentProposal[]>([]);
  const [unmatched, setUnmatched] = useState<string[]>([]);

  const refreshBatches = useCallback(async (documentSetId: number) => {
    const result = await listAmendmentBatches(documentSetId);
    setBatches(result);
    return result;
  }, []);

  const refreshProposals = useCallback(async (batchId: number) => {
    const result = await listAmendmentProposals(batchId);
    setProposals(result);
  }, []);

  useEffect(() => {
    if (!selectedDocumentSetId) return;
    void refreshBatches(Number(selectedDocumentSetId));
  }, [selectedDocumentSetId, refreshBatches]);

  useEffect(() => {
    if (selectedBatchId === null) {
      setProposals([]);
      return;
    }
    void refreshProposals(selectedBatchId);
  }, [selectedBatchId, refreshProposals]);

  const currentSourceIdentity = sourceIdentity(
    sourceMode,
    sourceUrl,
    sourceFile
  );
  const hasCurrentSourceExtraction =
    sourceMode === "text" ||
    (currentSourceIdentity !== null &&
      currentSourceIdentity === extractedSourceIdentity);
  const canAnalyze = Boolean(rawText.trim()) && hasCurrentSourceExtraction;

  const handleSourceModeChange = useCallback((mode: AmendmentSourceMode) => {
    setSourceMode(mode);
    setRawText("");
    setExtractedSourceIdentity(null);
  }, []);

  const handleExtract = useCallback(async () => {
    const identity = sourceIdentity(sourceMode, sourceUrl, sourceFile);
    if (!identity) {
      toast.error(
        sourceMode === "url"
          ? "Enter an amendment source URL."
          : "Choose a PDF file."
      );
      return;
    }

    setExtracting(true);
    try {
      const result =
        sourceMode === "url"
          ? await extractAmendmentUrl(sourceUrl.trim())
          : await extractAmendmentPdf(sourceFile as File);
      setRawText(result.text);
      setExtractedSourceIdentity(identity);
      toast.success(
        `Extracted text from ${result.display_name}. Review it before analysis.`
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Source extraction failed.");
    } finally {
      setExtracting(false);
    }
  }, [sourceFile, sourceMode, sourceUrl]);

  const handleAnalyze = useCallback(async () => {
    if (!selectedDocumentSetId || !canAnalyze) return;
    setAnalyzing(true);
    try {
      const result = await analyzeAmendment(
        Number(selectedDocumentSetId),
        rawText
      );
      toast.success(
        `Analyzed: ${result.proposals.length} proposal(s), ${result.unmatched_instructions.length} unmatched.`
      );
      setRawText("");
      await refreshBatches(Number(selectedDocumentSetId));
      setSelectedBatchId(result.batch.id);
      setProposals(result.proposals);
      setUnmatched(result.unmatched_instructions);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  }, [selectedDocumentSetId, rawText, canAnalyze, refreshBatches]);

  const selectedBatch = useMemo(
    () => batches.find((b) => b.id === selectedBatchId) ?? null,
    [batches, selectedBatchId]
  );

  return (
    <SettingsLayouts.Root width="lg">
      <SettingsLayouts.Header
        icon={SvgHistory}
        title="Updates"
        description="Paste an official amendment/update text scoped to a document set. It will be segmented into atomic changes, matched against the document set's existing chunks, and drafted for your review — nothing is written until you approve each proposal."
      />
      <SettingsLayouts.Body>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Text font="main-ui-action" color="text-04">
              Document Set
            </Text>
            <InputSelect
              value={selectedDocumentSetId ?? ""}
              onValueChange={(value) => {
                setSelectedDocumentSetId(value);
                setSelectedBatchId(null);
              }}
            >
              <InputSelect.Trigger />
              <InputSelect.Content>
                {documentSets.map((documentSet) => (
                  <InputSelect.Item
                    key={documentSet.id}
                    value={String(documentSet.id)}
                  >
                    {documentSet.name}
                  </InputSelect.Item>
                ))}
              </InputSelect.Content>
            </InputSelect>
          </div>

          {selectedDocumentSetId && (
            <>
              <div className="flex flex-col gap-2">
                <Text font="main-ui-action" color="text-04">
                  Amendment source
                </Text>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    prominence={sourceMode === "text" ? "primary" : "secondary"}
                    onClick={() => handleSourceModeChange("text")}
                  >
                    Text
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    prominence={sourceMode === "url" ? "primary" : "secondary"}
                    onClick={() => handleSourceModeChange("url")}
                  >
                    URL
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    prominence={sourceMode === "pdf" ? "primary" : "secondary"}
                    onClick={() => handleSourceModeChange("pdf")}
                  >
                    PDF
                  </Button>
                </div>

                {sourceMode === "url" && (
                  <div className="flex gap-2">
                    <InputTypeIn
                      aria-label="Amendment source URL"
                      value={sourceUrl}
                      onChange={(event) => {
                        setSourceUrl(event.target.value);
                        setRawText("");
                        setExtractedSourceIdentity(null);
                      }}
                      placeholder="https://www.resmigazete.gov.tr/..."
                    />
                    <Button
                      type="button"
                      onClick={() => void handleExtract()}
                      disabled={extracting || !sourceUrl.trim()}
                    >
                      {extracting ? "Extracting…" : "Extract"}
                    </Button>
                  </div>
                )}

                {sourceMode === "pdf" && (
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      ref={pdfInputRef}
                      aria-label="Amendment source PDF"
                      className="hidden"
                      type="file"
                      accept="application/pdf,.pdf"
                      onChange={(event) => {
                        setSourceFile(event.target.files?.[0] ?? null);
                        setRawText("");
                        setExtractedSourceIdentity(null);
                      }}
                    />
                    <Button
                      type="button"
                      prominence="secondary"
                      onClick={() => pdfInputRef.current?.click()}
                    >
                      {sourceFile ? "Choose another PDF" : "Choose PDF"}
                    </Button>
                    {sourceFile && (
                      <Text font="main-ui-body" color="text-03">
                        {sourceFile.name}
                      </Text>
                    )}
                    <Button
                      type="button"
                      onClick={() => void handleExtract()}
                      disabled={extracting || !sourceFile}
                    >
                      {extracting ? "Extracting…" : "Extract"}
                    </Button>
                  </div>
                )}

                <Text font="secondary-body" color="text-03">
                  {sourceMode === "text"
                    ? "Paste the official amendment/update text below."
                    : "Extracted text remains editable. Change the source and extract again before analysis."}
                </Text>
                <Text font="main-ui-action" color="text-04">
                  Amendment text
                </Text>
                <InputTextArea
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  rows={8}
                  autoResize
                  maxRows={20}
                  placeholder="Paste the official amendment/update text here..."
                />
                <div className="flex justify-end">
                  <Button
                    onClick={() => void handleAnalyze()}
                    disabled={analyzing || !canAnalyze}
                  >
                    {analyzing ? "Analyzing…" : "Analyze"}
                  </Button>
                </div>
              </div>

              {batches.length > 0 && (
                <div className="flex flex-col gap-2">
                  <Text font="main-ui-action" color="text-04">
                    Past batches
                  </Text>
                  <div className="flex flex-wrap gap-2">
                    {batches.map((batch) => (
                      <Button
                        key={batch.id}
                        prominence={
                          batch.id === selectedBatchId ? "primary" : "secondary"
                        }
                        size="sm"
                        onClick={() => setSelectedBatchId(batch.id)}
                      >
                        {`Batch #${batch.id} (${batch.status})`}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              {selectedBatch && (
                <div className="flex flex-col gap-3">
                  {selectedBatch.status === "failed" && (
                    <Text font="main-ui-body" color="text-05" as="p">
                      {`Analysis failed: ${selectedBatch.error_message ?? "unknown error"}`}
                    </Text>
                  )}

                  {unmatched.length > 0 && (
                    <div className="rounded-lg border border-border-02 p-3">
                      <Text font="main-ui-action" color="text-04">
                        Unmatched instructions
                      </Text>
                      {unmatched.map((instr, i) => (
                        <Text
                          key={i}
                          font="main-ui-body"
                          color="text-03"
                          as="p"
                        >
                          {instr}
                        </Text>
                      ))}
                    </div>
                  )}

                  {proposals.map((proposal) => (
                    <ProposalCard
                      key={proposal.id}
                      proposal={proposal}
                      onDecided={() => void refreshProposals(selectedBatch.id)}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </SettingsLayouts.Body>
    </SettingsLayouts.Root>
  );
}
