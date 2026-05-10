import { useMemo, useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Loader2, ScanText } from 'lucide-react';
import type { ChunkPreviewPayload } from '@/lib/types';

const MAX_CHUNKS_FOR_UI = 40;

interface Props {
  open: boolean;
  file: File | null;
  payload: ChunkPreviewPayload | null;
  onOpenChange: (open: boolean) => void;
  onIngestWithoutOcr: () => void;
  /** useLayoutReader: when true, include LayoutReader reading-order analysis */
  onRunPaddleOcr: (useLayoutReader: boolean) => void;
  paddleAvailable: boolean;
  layoutAnalysisAvailable: boolean;
  continuing?: boolean;
}

export default function ChunkPreviewModal({
  open,
  file,
  payload,
  onOpenChange,
  onIngestWithoutOcr,
  onRunPaddleOcr,
  paddleAvailable,
  layoutAnalysisAvailable,
  continuing,
}: Props) {
  const chunks = useMemo(() => payload?.chunks ?? [], [payload]);
  const limited = chunks.slice(0, MAX_CHUNKS_FOR_UI);
  const [useLayoutReader, setUseLayoutReader] = useState(false);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[980px] max-h-[90vh] flex flex-col gap-0 p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-6 pb-2 shrink-0">
          <DialogTitle>Chunk preview (no OCR)</DialogTitle>
          <DialogDescription>
            Review extracted text chunks. If the PDF is scanned or text looks incomplete, run OCR
            before ingesting to Neo4j.
          </DialogDescription>
        </DialogHeader>

        <div className="px-6 pb-2 flex-1 min-h-0 overflow-hidden">
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground mb-3">
            <span className="font-semibold text-primary/90">
              {payload?.extracted_pages ?? 0} page(s)
            </span>
            <span>{payload?.chunk_count_total ?? limited.length} total chunk(s)</span>
            {payload?.extracted_chars != null ? (
              <span>· {payload.extracted_chars.toLocaleString()} chars</span>
            ) : null}
            {file ? <span className="truncate">· {file.name}</span> : null}
          </div>

          <ScrollArea className="h-[min(52vh,480px)] rounded-md border border-border bg-muted/20">
            <div className="p-4 space-y-3">
              {limited.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No text extracted — this PDF may be scanned. Run OCR below.
                </p>
              ) : (
                limited.map((c) => (
                  <div key={c.index} className="rounded-sm border border-border bg-background/50 p-3">
                    <div className="text-[11px] font-semibold text-primary/90 mb-1">
                      Chunk #{c.index}
                    </div>
                    <div className="text-[12px] leading-relaxed whitespace-pre-wrap break-words">
                      {c.text}
                    </div>
                  </div>
                ))
              )}
              {chunks.length > MAX_CHUNKS_FOR_UI ? (
                <p className="text-xs text-muted-foreground">
                  Showing first {MAX_CHUNKS_FOR_UI} of {chunks.length} chunks.
                </p>
              ) : null}
            </div>
          </ScrollArea>

          {/* Paddle OCR options */}
          {paddleAvailable && (
            <div className="mt-3 rounded-md border border-border bg-muted/30 px-4 py-3 flex items-start gap-3">
              <ScanText className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
              <div className="flex-1 min-w-0">
                <p className="text-[12px] font-medium text-foreground mb-1">Paddle OCR options</p>
                <label className="flex items-center gap-2 text-[11px] text-muted-foreground cursor-pointer select-none">
                  <input
                    type="checkbox"
                    name="use-layout-reader"
                    className="accent-primary"
                    checked={useLayoutReader}
                    disabled={!layoutAnalysisAvailable || continuing}
                    onChange={(e) => setUseLayoutReader(e.target.checked)}
                  />
                  <span>
                    Include LayoutReader reading order
                    {!layoutAnalysisAvailable && (
                      <span className="ml-1 text-destructive/70">
                        (requires <code className="font-mono">torch</code> +{' '}
                        <code className="font-mono">transformers</code>)
                      </span>
                    )}
                  </span>
                </label>
                {useLayoutReader && layoutAnalysisAvailable && (
                  <p className="mt-1 text-[10px] text-amber-600 dark:text-amber-400">
                    First run downloads hantian/layoutreader (~600 MB). Subsequent runs use the
                    cached model.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        <DialogFooter className="px-6 py-4 border-t border-border gap-2 sm:gap-2 flex-wrap">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={continuing}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={onIngestWithoutOcr}
            disabled={continuing}
          >
            {continuing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Ingesting…
              </>
            ) : (
              'Ingest without OCR'
            )}
          </Button>
          <Button
            type="button"
            variant="default"
            onClick={() => onRunPaddleOcr(useLayoutReader)}
            disabled={continuing || !paddleAvailable}
          >
            Run Paddle OCR
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
