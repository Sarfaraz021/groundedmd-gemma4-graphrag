import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import type { PDFDocumentProxy } from 'pdfjs-dist';
import { Loader2, ChevronLeft, ChevronRight, Sparkles } from 'lucide-react';
import DOMPurify from 'dompurify';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { AssistantMarkdown } from '@/components/AssistantMarkdown';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { ParsePreviewPayload } from '@/lib/types';

GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.mjs', import.meta.url).toString();

/** Avoid freezing the UI / OOM from huge OCR payloads in the markdown tab. */
const MAX_MARKDOWN_PREVIEW_CHARS = 350_000;
/** Cap SVG overlay rects (some PDFs return thousands of regions). */
const MAX_BOXES_PER_PAGE = 600;
/** Cap chunk merge work when building grounding from ``chunks[]``. */
const MAX_CHUNKS_FOR_GROUNDING = 4_000;
/** Avoid scanning 100k+ grounding keys from pathological API responses. */
const MAX_GROUNDING_KEYS_TO_SCAN = 25_000;

const CHUNK_TYPE_COLORS: Record<string, string> = {
  chunkText: 'rgba(40, 167, 69, 0.7)',
  chunkTable: 'rgba(0, 123, 255, 0.7)',
  chunkMarginalia: 'rgba(111, 66, 193, 0.7)',
  chunkFigure: 'rgba(255, 0, 255, 0.65)',
  chunkLogo: 'rgba(144, 238, 144, 0.8)',
  chunkCard: 'rgba(255, 165, 0, 0.75)',
  chunkAttestation: 'rgba(0, 255, 255, 0.7)',
  chunkScanCode: 'rgba(255, 193, 7, 0.8)',
  chunkForm: 'rgba(220, 20, 60, 0.7)',
  tableCell: 'rgba(173, 216, 230, 0.6)',
  table: 'rgba(70, 130, 180, 0.7)',
};

/**
 * Sanitize mixed markdown+HTML from OCR output before passing to
 * ReactMarkdown + rehype-raw.  Allows table structure and common formatting tags
 * while stripping scripts, event handlers, and unsafe attributes.
 */
function sanitizeOcrContent(raw: string): string {
  return DOMPurify.sanitize(raw, {
    ALLOWED_TAGS: [
      'p', 'br', 'b', 'i', 'strong', 'em', 'u', 's', 'del', 'ins',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption',
      'blockquote', 'pre', 'code', 'span', 'div', 'hr', 'a',
    ],
    ALLOWED_ATTR: ['href', 'title', 'id', 'colspan', 'rowspan', 'scope'],
  });
}

const ocrProse =
  'prose prose-sm dark:prose-invert max-w-none text-foreground ' +
  'prose-p:my-2 prose-headings:mt-4 prose-headings:mb-2 prose-headings:font-semibold ' +
  'prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-strong:text-foreground ' +
  'prose-table:w-full prose-table:border-collapse ' +
  '[&_table]:border [&_td]:border [&_th]:border [&_td]:px-2 [&_td]:py-1 [&_th]:px-2 [&_th]:py-1';

interface Props {
  open: boolean;
  file: File | null;
  payload: ParsePreviewPayload | null;
  onOpenChange: (open: boolean) => void;
  onContinue: () => void;
  continuing?: boolean;
  title?: string;
  groundingSourceLabel?: string;
}

/** Prefer top-level ``grounding``; otherwise merge ``chunks[].grounding`` (payload shapes vary). */
function mergedGrounding(payload: ParsePreviewPayload | null): Record<string, unknown> | undefined {
  if (!payload) return undefined;
  const top = payload.grounding;
  if (top && typeof top === 'object' && Object.keys(top).length > 0) {
    return top as Record<string, unknown>;
  }
  const chunks = payload.chunks;
  if (!Array.isArray(chunks)) return undefined;
  const merged: Record<string, unknown> = {};
  const limit = Math.min(chunks.length, MAX_CHUNKS_FOR_GROUNDING);
  for (let i = 0; i < limit; i++) {
    const raw = chunks[i];
    if (!raw || typeof raw !== 'object') continue;
    const c = raw as Record<string, unknown>;
    const gid = String(c.id ?? c.chunk_id ?? '');
    const g = c.grounding;
    if (gid && g && typeof g === 'object') merged[gid] = g;
  }
  return Object.keys(merged).length > 0 ? merged : undefined;
}

/** If all pages are ≥1, treat API as 1-based and shift to 0-based for pdf.js. */
function groundingPageOffset(grounding: Record<string, unknown>): number {
  const pages: number[] = [];
  let n = 0;
  for (const raw of Object.values(grounding)) {
    if (n++ >= 400) break;
    if (!raw || typeof raw !== 'object') continue;
    const g = raw as Record<string, unknown>;
    const p = typeof g.page === 'number' ? g.page : g.page != null ? Number(g.page) : NaN;
    if (Number.isFinite(p)) pages.push(p);
  }
  if (pages.length === 0) return 0;
  return Math.min(...pages) >= 1 ? -1 : 0;
}

function pageBoxForGrounding(
  grounding: Record<string, unknown> | undefined,
  pageZero: number
): Array<{
  id: string;
  left: number;
  top: number;
  width: number;
  height: number;
  color: string;
  label: string;
  readingRank?: number;
  text?: string;
}> {
  if (!grounding || typeof grounding !== 'object') return [];
  const pageOffset = groundingPageOffset(grounding);
  const out: Array<{
    id: string;
    left: number;
    top: number;
    width: number;
    height: number;
    color: string;
    label: string;
    readingRank?: number;
    text?: string;
  }> = [];

  let scanned = 0;
  for (const id of Object.keys(grounding)) {
    if (scanned++ >= MAX_GROUNDING_KEYS_TO_SCAN) break;
    const raw = grounding[id];
    if (!raw || typeof raw !== 'object') continue;
    const g = raw as Record<string, unknown>;
    const pageRaw = typeof g.page === 'number' ? g.page : g.page != null ? Number(g.page) : NaN;
    const page = Number.isFinite(pageRaw) ? pageRaw + pageOffset : NaN;
    if (Number.isFinite(page) && page !== pageZero) continue;
    if (!Number.isFinite(page) && pageZero !== 0) continue;

    const box = g.box as Record<string, unknown> | undefined;
    if (!box) continue;
    const left = Number(box.left);
    const top = Number(box.top);
    const right = Number(box.right);
    const bottom = Number(box.bottom);
    if (![left, top, right, bottom].every((n) => Number.isFinite(n))) continue;

    const t = typeof g.type === 'string' ? g.type : 'chunk';
    const color = CHUNK_TYPE_COLORS[t] ?? 'rgba(128, 128, 128, 0.65)';
    const readingRank = typeof g.reading_rank === 'number' ? g.reading_rank : undefined;
    const text = typeof g.text === 'string' ? g.text : undefined;
    out.push({
      id,
      left,
      top,
      width: right - left,
      height: bottom - top,
      color,
      label: `${t}:${id.slice(0, 8)}`,
      readingRank,
      text,
    });
    if (out.length >= MAX_BOXES_PER_PAGE) break;
  }
  return out;
}

export default function ParsePreviewModal({
  open,
  file,
  payload,
  onOpenChange,
  onContinue,
  continuing,
  title,
  groundingSourceLabel,
}: Props) {
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [loadingPdf, setLoadingPdf] = useState(false);
  const [pageIndex, setPageIndex] = useState(0);
  const [activeTab, setActiveTab] = useState('html');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [viewportSize, setViewportSize] = useState({ w: 0, h: 0 });
  const renderSeq = useRef(0);

  // Structured view state
  const [structuredMarkdown, setStructuredMarkdown] = useState<string | null>(null);
  const [structuringLoading, setStructuringLoading] = useState(false);
  const [structuringError, setStructuringError] = useState<string | null>(null);

  const { ocrContent, htmlTruncated } = useMemo(() => {
    const raw = payload?.markdown ?? '';
    if (raw.length <= MAX_MARKDOWN_PREVIEW_CHARS) {
      return { ocrContent: sanitizeOcrContent(raw), htmlTruncated: false };
    }
    return {
      ocrContent: sanitizeOcrContent(raw.slice(0, MAX_MARKDOWN_PREVIEW_CHARS)),
      htmlTruncated: true,
    };
  }, [payload]);

  const grounding = useMemo(() => mergedGrounding(payload), [payload]);

  const numPages = pdf?.numPages ?? 0;

  useEffect(() => {
    if (!open || !file) {
      setPdf(null);
      setPdfError(null);
      setPageIndex(0);
      return;
    }
    let cancelled = false;
    setLoadingPdf(true);
    setPdfError(null);
    void (async () => {
      try {
        const buf = await file.arrayBuffer();
        const doc = await getDocument({ data: buf }).promise;
        if (!cancelled) {
          setPdf(doc);
          setPageIndex(0);
        }
      } catch (e) {
        if (!cancelled) {
          setPdfError(e instanceof Error ? e.message : 'Failed to load PDF');
          setPdf(null);
        }
      } finally {
        if (!cancelled) setLoadingPdf(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, file]);

  useEffect(() => {
    if (!open || !pdf || activeTab !== 'layout') return;
    const id = ++renderSeq.current;
    setPdfError(null);
    void (async () => {
      const canvas = canvasRef.current;
      const wrap = wrapRef.current;
      if (!canvas || !wrap) return;
      try {
        const page = await pdf.getPage(pageIndex + 1);
        if (id !== renderSeq.current) return;
        const base = page.getViewport({ scale: 1 });
        if (!base.width || !base.height) {
          if (id === renderSeq.current) setPdfError('Invalid PDF page dimensions.');
          return;
        }
        const maxW = Math.min(wrap.clientWidth || 640, 900);
        const scale = maxW / base.width;
        if (!Number.isFinite(scale) || scale <= 0) {
          if (id === renderSeq.current) setPdfError('Could not compute PDF scale.');
          return;
        }
        const viewport = page.getViewport({ scale });
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          if (id === renderSeq.current) setPdfError('Canvas is unavailable.');
          return;
        }
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: ctx, viewport }).promise;
        if (id !== renderSeq.current) return;
        setViewportSize({ w: viewport.width, h: viewport.height });
      } catch (e) {
        if (id !== renderSeq.current) return;
        setPdfError(e instanceof Error ? e.message : 'PDF page render failed.');
        setViewportSize({ w: 0, h: 0 });
      }
    })();
  }, [open, pdf, pageIndex, activeTab]);

  const boxes = useMemo(
    () => pageBoxForGrounding(grounding, pageIndex).map((b) => ({
      ...b,
      px: {
        x: b.left * viewportSize.w,
        y: b.top * viewportSize.h,
        w: b.width * viewportSize.w,
        h: b.height * viewportSize.h,
      },
    })),
    [grounding, pageIndex, viewportSize.w, viewportSize.h]
  );

  useEffect(() => {
    if (!open) {
      setPageIndex(0);
      setActiveTab('html');
      setViewportSize({ w: 0, h: 0 });
      setStructuredMarkdown(null);
      setStructuringError(null);
      setStructuringLoading(false);
    }
  }, [open]);

  const handleStructuredView = useCallback(async () => {
    if (!payload?.markdown) return;
    setStructuringLoading(true);
    setStructuringError(null);
    try {
      const form = new FormData();
      form.append('raw_markdown', payload.markdown);
      const res = await fetch('/api/ingest/restructure', { method: 'POST', body: form });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const data = await res.json() as { markdown: string };
      setStructuredMarkdown(data.markdown);
      setActiveTab('structured');
    } catch (e) {
      setStructuringError(e instanceof Error ? e.message : 'Restructuring failed.');
    } finally {
      setStructuringLoading(false);
    }
  }, [payload]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[940px] max-h-[90vh] flex flex-col gap-0 p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-6 pb-2 shrink-0">
          <DialogTitle>{title ?? 'OCR preview'}</DialogTitle>
          <DialogDescription>
            Review the parsed HTML output and layout boxes. Use "Structured View" to run LLM formatting, then continue to graph ingest.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="html" value={activeTab} onValueChange={setActiveTab} className="flex-1 min-h-0 flex flex-col px-6 pb-2">
          <TabsList className="w-fit">
            <TabsTrigger value="html">HTML</TabsTrigger>
            {structuredMarkdown && <TabsTrigger value="structured">Structured</TabsTrigger>}
            <TabsTrigger value="layout">Layout & boxes</TabsTrigger>
          </TabsList>

          {/* Raw HTML tab */}
          <TabsContent value="html" className="flex-1 min-h-0 mt-3 data-[state=inactive]:hidden">
            <ScrollArea className="h-[min(52vh,420px)] rounded-md border border-border bg-muted/20">
              <div className="p-4 text-sm leading-relaxed">
                {htmlTruncated && (
                  <p className="mb-3 text-xs text-amber-600 dark:text-amber-400 border border-amber-500/30 rounded px-2 py-1.5 bg-amber-500/10">
                    Preview truncated to {MAX_MARKDOWN_PREVIEW_CHARS.toLocaleString()} characters. Full text is still used
                    when you continue to ingest.
                  </p>
                )}
                {ocrContent ? (
                  <div className={ocrProse}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
                      {ocrContent}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="text-muted-foreground">No output returned.</p>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* LLM-structured tab — only visible after the user requests it */}
          {structuredMarkdown && (
            <TabsContent value="structured" className="flex-1 min-h-0 mt-3 data-[state=inactive]:hidden">
              <ScrollArea className="h-[min(52vh,420px)] rounded-md border border-border bg-muted/20">
                <div className="p-4 text-sm leading-relaxed">
                  <AssistantMarkdown content={structuredMarkdown} />
                </div>
              </ScrollArea>
            </TabsContent>
          )}

          {/* Layout & boxes tab */}
          <TabsContent value="layout" className="flex-1 min-h-0 mt-3 data-[state=inactive]:hidden">
            <div className="space-y-3">
              {loadingPdf && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading PDF…
                </div>
              )}
              {pdfError && <p className="text-sm text-destructive">{pdfError}</p>}
              {numPages > 0 && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    disabled={pageIndex <= 0}
                    onClick={() => setPageIndex((p) => Math.max(0, p - 1))}
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </Button>
                  <span>
                    Page {pageIndex + 1} / {numPages}
                  </span>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    disabled={pageIndex >= numPages - 1}
                    onClick={() => setPageIndex((p) => Math.min(numPages - 1, p + 1))}
                  >
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                  <span className="ml-2">{boxes.length} region(s) on this page</span>
                </div>
              )}
              <div
                ref={wrapRef}
                className="relative rounded-md border border-border bg-background overflow-auto max-h-[min(52vh,420px)] flex justify-center"
              >
                <div
                  className="relative inline-block"
                  style={
                    viewportSize.w > 0
                      ? { width: viewportSize.w, height: viewportSize.h }
                      : undefined
                  }
                >
                  <canvas ref={canvasRef} className="block max-w-full h-auto" />
                  {numPages > 0 && viewportSize.w > 0 && (
                    <svg
                      className="absolute left-0 top-0 pointer-events-none"
                      width={viewportSize.w}
                      height={viewportSize.h}
                    >
                      {boxes.map((b) => (
                        <g key={b.id}>
                          <rect
                            x={b.px.x}
                            y={b.px.y}
                            width={b.px.w}
                            height={b.px.h}
                            fill="none"
                            stroke={b.color}
                            strokeWidth={2}
                          />
                          {b.readingRank != null && (
                            <text
                              x={b.px.x + 3}
                              y={b.px.y + 11}
                              fontSize="9"
                              fontWeight="bold"
                              fill={b.color}
                              stroke="white"
                              strokeWidth="2.5"
                              paintOrder="stroke"
                            >
                              {b.readingRank + 1}
                            </text>
                          )}
                        </g>
                      ))}
                    </svg>
                  )}
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                Boxes use normalized coordinates for PDF overlay (page index is 0-based).{' '}
                {groundingSourceLabel ?? 'Colors follow chunk types.'}{' '}
                {boxes.some((b) => b.readingRank != null) && (
                  <span className="font-medium text-primary/80">
                    Numbers show LayoutReader reading order (1 = first).
                  </span>
                )}
              </p>
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter className="px-6 py-4 border-t border-border gap-2 sm:gap-2 flex-wrap">
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={continuing}>
            Cancel
          </Button>

          {/* Structured View button — triggers LLM call on demand */}
          <Button
            type="button"
            variant="secondary"
            onClick={handleStructuredView}
            disabled={structuringLoading || !payload?.markdown || continuing}
          >
            {structuringLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Structuring…
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                {structuredMarkdown ? 'Re-run Structured View' : 'Structured View'}
              </>
            )}
          </Button>

          {structuringError && (
            <p className="text-xs text-destructive w-full text-right">{structuringError}</p>
          )}

          <Button type="button" onClick={onContinue} disabled={continuing || !payload?.markdown}>
            {continuing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Ingesting…
              </>
            ) : (
              'Continue to graph ingest'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
