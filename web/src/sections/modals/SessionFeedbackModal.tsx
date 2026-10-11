"use client";

import { useEffect, useRef, useState } from "react";
import { Button, InputTextArea, Modal, Text } from "@opal/components";
import { InputVertical, toast } from "@opal/layouts";
import { SvgStar } from "@opal/icons";
import type { IconProps } from "@opal/types";
import type {
  BackendChatSession,
  SessionFeedbackRating,
  SessionFeedbackResponse,
} from "@/app/app/interfaces";
import { useChatSessionStore } from "@/app/app/stores/useChatSessionStore";
import { getLatestMessageChain } from "@/app/app/services/messageTree";
import {
  SESSION_FEEDBACK_MAX_TEXT_LENGTH,
  SESSION_FEEDBACK_OPTIONS,
} from "@/lib/chat/session-feedback/constants";

interface SessionFeedbackModalProps {
  sessionId: string;
  onClose: () => void;
}

function FilledStar(props: IconProps) {
  return <SvgStar {...props} fill="currentColor" />;
}

export default function SessionFeedbackModal({
  sessionId,
  onClose,
}: SessionFeedbackModalProps) {
  const session = useChatSessionStore((state) => state.sessions.get(sessionId));
  const feedback = session?.sessionFeedback;
  const [rating, setRating] = useState<SessionFeedbackRating | null>(
    feedback?.rating ?? null
  );
  const [feedbackText, setFeedbackText] = useState(
    feedback?.feedback_text ?? ""
  );
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const expectedVersion = useRef(feedback?.version ?? 0);
  const submitting = useRef(false);
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const ratingRefs = useRef<
    Partial<Record<SessionFeedbackRating, HTMLElement | null>>
  >({});

  const latestAssistantMessage = session
    ? getLatestMessageChain(session.messageTree).findLast(
        (message) => message.type === "assistant" && !message.is_generating
      )
    : undefined;
  const hasNewAnswers =
    feedback &&
    latestAssistantMessage?.messageId != null &&
    latestAssistantMessage.messageId !== feedback.evaluated_through_message_id;

  function close() {
    if (mounted.current && !submitting.current) onClose();
  }

  async function refreshFeedback() {
    const response = await fetch(`/api/chat/get-chat-session/${sessionId}`);
    if (!response.ok) throw new Error("Session feedback refresh failed");
    const latest = (await response.json()) as BackendChatSession;
    const latestFeedback = latest.session_feedback ?? null;
    useChatSessionStore
      .getState()
      .updateSessionFeedback(sessionId, latestFeedback);
    expectedVersion.current =
      useChatSessionStore.getState().sessions.get(sessionId)?.sessionFeedback
        ?.version ?? 0;
  }

  async function handleSubmit() {
    if (!rating || submitting.current) return;
    submitting.current = true;
    setIsSubmitting(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/chat/chat-session/${sessionId}/feedback`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rating,
            feedback_text: feedbackText.trim() || null,
            expected_version: expectedVersion.current,
          }),
        }
      );
      if (response.status === 409) {
        if (!mounted.current) return;
        await refreshFeedback();
        if (!mounted.current) return;
        setError(
          "Değerlendirme başka bir yerde güncellendi. Güncel puan aşağıda gösteriliyor; taslağınız korundu. Kaydetmeden önce kontrol edin."
        );
        return;
      }
      if (!response.ok) {
        if (!mounted.current) return;
        setError(
          "Değerlendirme kaydedilemedi. Taslağınızı koruyarak tekrar deneyebilirsiniz."
        );
        return;
      }
      const saved = (await response.json()) as SessionFeedbackResponse;
      useChatSessionStore.getState().updateSessionFeedback(sessionId, saved);
      if (!mounted.current) return;
      expectedVersion.current = saved.version;
      toast.success("Sohbet değerlendirmesi kaydedildi.");
      onClose();
    } catch {
      if (!mounted.current) return;
      setError(
        "Değerlendirme kaydedilemedi. Bağlantınızı kontrol edip tekrar deneyin; taslağınız korundu."
      );
    } finally {
      submitting.current = false;
      if (mounted.current) setIsSubmitting(false);
    }
  }

  return (
    <Modal open onOpenChange={(open) => !open && close()}>
      <Modal.Content width="sm">
        <Modal.Header
          icon={SvgStar}
          title="Sohbeti değerlendir"
          description="Bu sohbetin tamamı için bir puan seçin."
          onClose={close}
        />
        <Modal.Body>
          <div
            role="radiogroup"
            aria-label="Sohbet puanı"
            className="grid grid-cols-4 gap-2"
          >
            {SESSION_FEEDBACK_OPTIONS.map((option) => (
              <div
                key={option.rating}
                className="flex flex-col items-center gap-2"
              >
                <Button
                  ref={(element) => {
                    ratingRefs.current[option.rating] = element;
                  }}
                  icon={
                    rating && option.rating <= rating ? FilledStar : SvgStar
                  }
                  prominence={
                    rating === option.rating ? "primary" : "secondary"
                  }
                  role="radio"
                  aria-checked={rating === option.rating}
                  aria-label={`${option.rating} yıldız: ${option.label}`}
                  tabIndex={
                    rating === option.rating || (!rating && option.rating === 1)
                      ? 0
                      : -1
                  }
                  disabled={isSubmitting}
                  onClick={() => setRating(option.rating)}
                  onKeyDown={(event) => {
                    const direction =
                      event.key === "ArrowRight" || event.key === "ArrowDown"
                        ? 1
                        : event.key === "ArrowLeft" || event.key === "ArrowUp"
                          ? -1
                          : 0;
                    if (!direction || isSubmitting) return;
                    event.preventDefault();
                    const next =
                      SESSION_FEEDBACK_OPTIONS[
                        (option.rating -
                          1 +
                          direction +
                          SESSION_FEEDBACK_OPTIONS.length) %
                          SESSION_FEEDBACK_OPTIONS.length
                      ]!;
                    setRating(next.rating);
                    ratingRefs.current[next.rating]?.focus();
                  }}
                />
                <Text font="secondary-body" color="text-03">
                  {option.label}
                </Text>
              </div>
            ))}
          </div>
          {rating && (
            <Text font="main-ui-action" aria-live="polite">
              {`${rating}/4 — ${SESSION_FEEDBACK_OPTIONS.find((option) => option.rating === rating)!.label}`}
            </Text>
          )}
          <InputVertical
            withLabel="session-feedback-text"
            title="Açıklama"
            suffix="opsiyonel"
          >
            <InputTextArea
              id="session-feedback-text"
              aria-label="Açıklama (opsiyonel)"
              placeholder="Neyin yanlış, eksik veya doğru olduğunu anlatabilirsiniz."
              value={feedbackText}
              onChange={(event) => setFeedbackText(event.target.value)}
              maxLength={SESSION_FEEDBACK_MAX_TEXT_LENGTH}
              variant={isSubmitting ? "disabled" : "primary"}
            />
          </InputVertical>
          <Text
            font="secondary-body"
            color="text-03"
          >{`${feedbackText.length}/${SESSION_FEEDBACK_MAX_TEXT_LENGTH}`}</Text>
          {feedback && (
            <div className="flex flex-col gap-1">
              <Text font="secondary-body" color="text-03">
                {`Kayıtlı puan: ${feedback.rating}/4 — ${feedback.label}. Son değerlendirme: ${new Date(feedback.time_updated).toLocaleString("tr-TR")}.`}
              </Text>
              <Text font="secondary-body" color="text-03">
                {hasNewAnswers
                  ? "Bu değerlendirmeden sonra sohbete yeni yanıtlar eklendi. Kaydetmek puanı mevcut sohbete uygular."
                  : "Bu puan, kaydedildiği andaki sohbet içindir."}
              </Text>
            </div>
          )}
          {error && (
            <Text role="alert" color="text-03">
              {error}
            </Text>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button
            prominence="secondary"
            disabled={isSubmitting}
            onClick={close}
          >
            Vazgeç
          </Button>
          <Button disabled={!rating || isSubmitting} onClick={handleSubmit}>
            {isSubmitting ? "Kaydediliyor…" : "Kaydet"}
          </Button>
        </Modal.Footer>
      </Modal.Content>
    </Modal>
  );
}
