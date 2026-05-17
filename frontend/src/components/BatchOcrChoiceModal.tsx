import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';

export type BatchOcrMode = 'none' | 'docling';

interface Props {
  open: boolean;
  fileCount: number;
  pdfCount: number;
  paddleAvailable: boolean;
  onConfirm: (mode: BatchOcrMode) => void;
  onCancel: () => void;
}

const MODES: { value: BatchOcrMode; label: string; description: string }[] = [
  {
    value: 'none',
    label: 'No OCR',
    description: 'Fast raw text extraction. Best for text-native PDFs.',
  },
  {
    value: 'docling',
    label: 'Docling OCR (EC2)',
    description: 'AI-powered document understanding — tables, diagrams, and complex layouts. Runs on GPU.',
  },
];

export default function BatchOcrChoiceModal({
  open,
  fileCount,
  pdfCount,
  paddleAvailable,
  onConfirm,
  onCancel,
}: Props) {
  const [selected, setSelected] = useState<BatchOcrMode>('none');

  const availableModes = MODES.filter((m) => {
    if (m.value === 'docling') return paddleAvailable && pdfCount > 0;
    return true;
  });

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onCancel(); }}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="text-sm font-semibold uppercase tracking-[0.1em]">
            Batch ingest — PDF processing
          </DialogTitle>
          <DialogDescription className="text-xs text-muted-foreground pt-1">
            {fileCount} file{fileCount !== 1 ? 's' : ''} selected
            {pdfCount > 0 ? ` (${pdfCount} PDF${pdfCount !== 1 ? 's' : ''})` : ''}.
            Choose how PDFs should be processed before graph ingestion.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2 py-1">
          {availableModes.map((mode) => (
            <button
              key={mode.value}
              type="button"
              onClick={() => setSelected(mode.value)}
              className={`w-full text-left px-3 py-2.5 border transition-colors ${
                selected === mode.value
                  ? 'border-primary bg-primary/5 text-foreground'
                  : 'border-border bg-card text-muted-foreground hover:border-primary/40 hover:text-foreground'
              }`}
            >
              <div className="text-[12px] font-semibold uppercase tracking-[0.08em] mb-0.5">
                {mode.label}
              </div>
              <div className="text-[11px] leading-relaxed">{mode.description}</div>
            </button>
          ))}
        </div>

        <DialogFooter className="gap-2 pt-1">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] border border-border text-muted-foreground hover:text-foreground transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onConfirm(selected)}
            className="px-4 py-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] bg-primary text-primary-foreground hover:opacity-90 transition-all"
          >
            Start ingest
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
