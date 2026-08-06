"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  Button,
  InputSelect,
  InputTextArea,
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
  listAmendmentBatches,
  listAmendmentProposals,
  rejectProposal,
} from "@/lib/regulatory/amendments";

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

  const handleAnalyze = useCallback(async () => {
    if (!selectedDocumentSetId || !rawText.trim()) return;
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
  }, [selectedDocumentSetId, rawText, refreshBatches]);

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
              value={selectedDocumentSetId ?? undefined}
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
                  Paste amendment text
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
                    disabled={analyzing || !rawText.trim()}
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
