import {useState} from 'react';
import type {FormEvent} from 'react';
import {Modal} from './Modal';
import {Button} from './Button';
import {TextField} from './TextField';

export function NamePromptModal({
  title,
  label,
  initialValue = '',
  confirmLabel = 'Save',
  onSubmit,
  onCancel,
}: {
  title: string;
  label: string;
  initialValue?: string;
  confirmLabel?: string;
  onSubmit: (value: string) => Promise<void>;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initialValue);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!value.trim()) return;
    setSubmitting(true);
    try {
      await onSubmit(value.trim());
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Modal title={title} onClose={onCancel}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <TextField
          label={label}
          autoFocus
          value={value}
          onChange={e => setValue(e.target.value)}
        />
        <div className="flex justify-end gap-2">
          <Button type="button" variant="secondary" onClick={onCancel} disabled={submitting}>
            Cancel
          </Button>
          <Button type="submit" disabled={submitting || !value.trim()}>
            {submitting ? 'Saving…' : confirmLabel}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
