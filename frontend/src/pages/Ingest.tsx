import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { API_BASE } from '@/lib/api';
import { readSseJsonLines } from '@/lib/sse';
import ChunkPreviewModal from '@/components/ChunkPreviewModal';
import ParsePreviewModal from '@/components/ParsePreviewModal';
import PreviewErrorBoundary from '@/components/PreviewErrorBoundary';
import {
  CheckCircle2,
  Loader2,
  AlertCircle,
  Upload,
  ArrowLeft,
  ChevronDown,
  FileText,
  Brain,
  Database,
} from 'lucide-react';
import type {
  ChunkPreviewPayload,
  ParsePreviewPayload,
  IngestStreamEvent,
  IngestMeta,
  GraphPreview,
} from '@/lib/types';

// ── Pre-loaded knowledge base ────────────────────────────────────────────────
const PUBLICATIONS = [
  {
    title: 'TBI Classification & Nomenclature',
    file: 'TBI-classification-nomenclature-workshop-agenda-508c.pdf',
    description: 'NINDS workshop establishing standardized terminology and severity classification for traumatic brain injury — the foundation for consistent clinical documentation.',
  },
  {
    title: 'Blood-Based Biomarkers for TBI',
    file: 'blood-based-biomarkers-tbi-wg-508c.pdf',
    description: 'Working group report on serum and plasma biomarkers (GFAP, UCH-L1, S100B, NfL) for diagnosis, prognosis, and monitoring of TBI at the point of care.',
  },
  {
    title: 'Clinical Symptoms Days 1–14',
    file: 'clinical-symptoms-days1-14-wg-508c.pdf',
    description: 'Systematic review of symptom trajectories in the acute phase — headache, cognitive deficits, sleep disturbance — with recommended assessment instruments.',
  },
  {
    title: 'Knowledge to Practice',
    file: 'knowlege-to-practice-wg-508c.pdf',
    description: 'Translational framework for moving TBI research findings into frontline clinical workflows, particularly in under-resourced and rural settings.',
  },
  {
    title: 'Psychosocial & Environmental Modifiers',
    file: 'psychosocial-environmental-modifiers-wg-508c.pdf',
    description: 'Evidence synthesis on how social determinants — access to care, mental health history, family support — modulate recovery outcomes after TBI.',
  },
  {
    title: 'Retrospective Classification of TBI',
    file: 'retrospective-classification-tbi-wg-508c.pdf',
    description: 'Methods for classifying TBI severity and type from existing clinical records, enabling research on populations where prospective data collection was not possible.',
  },
  {
    title: 'TBI Neuroimaging Biomarkers',
    file: 'tbi-imaging-wg-508c.pdf',
    description: 'CT and MRI imaging protocols, lesion classification systems, and quantitative biomarkers for structural TBI assessment — including resource-limited imaging guidance.',
  },
];

// ── Local job state (no session dependency) ──────────────────────────────────
interface IngestJob {
  id: string;
  fileName: string;
  status: 'running' | 'complete' | 'error';
  events: IngestStreamEvent[];
  meta?: IngestMeta;
  error?: string;
  graphPreview?: GraphPreview;
}

function normalizeEvent(raw: Record<string, unknown>): IngestStreamEvent {
  return {
    event_type: String(raw.event_type ?? ''),
    task_name: raw.task_name != null ? String(raw.task_name) : undefined,
    message: raw.message != null ? String(raw.message) : undefined,
    payload: raw.payload as Record<string, unknown> | undefined,
    timestamp: raw.timestamp != null ? String(raw.timestamp) : undefined,
    hint: raw.hint as IngestStreamEvent['hint'],
  };
}

// ── Component ────────────────────────────────────────────────────────────────
export default function Ingest() {
  const navigate = useNavigate();
  const dropRef = useRef<HTMLDivElement>(null);

  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [paddleAvailable, setPaddleAvailable] = useState(false);
  const [layoutAvailable, setLayoutAvailable] = useState(false);

  const [chunkPreview, setChunkPreview] = useState<{
    file: File; payload: ChunkPreviewPayload;
  } | null>(null);
  const [parsePreview, setParsePreview] = useState<{
    file: File;
    payload: ParsePreviewPayload;
  } | null>(null);
  const [continuingPreview, setContinuingPreview] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/ingest/config`)
      .then(r => r.json())
      .then((x: Record<string, unknown>) => {
        setPaddleAvailable(Boolean(x.paddle_ocr_preview));
        setLayoutAvailable(Boolean(x.layout_analysis));
      })
      .catch(() => {});
  }, []);

  // ── SSE runner ──────────────────────────────────────────────────────────────
  const runSse = useCallback(async (jobId: string, res: Response) => {
    let finished = false;
    const finish = (payload: { status: 'complete' | 'error'; error?: string; graphPreview?: GraphPreview }) => {
      if (finished) return;
      finished = true;
      setJobs(prev => prev.map(j => j.id === jobId ? { ...j, ...payload } : j));
    };
    try {
      await readSseJsonLines(res, (data) => {
        const t = data.type as string;
        if (t === 'meta') {
          setJobs(prev => prev.map(j => j.id === jobId ? { ...j, meta: data as unknown as IngestMeta } : j));
        } else if (t === 'event' && data.event && typeof data.event === 'object') {
          setJobs(prev => prev.map(j =>
            j.id === jobId
              ? { ...j, events: [...j.events, normalizeEvent(data.event as Record<string, unknown>)] }
              : j
          ));
        } else if (t === 'done') {
          finish({ status: 'complete', graphPreview: data.graph_preview as GraphPreview | undefined });
        } else if (t === 'error') {
          finish({ status: 'error', error: String(data.message ?? 'Ingest failed') });
        }
      });
      if (!finished) finish({ status: 'complete' });
    } catch (e) {
      finish({ status: 'error', error: e instanceof Error ? e.message : 'Network error' });
    }
  }, []);

  // ── File handler ────────────────────────────────────────────────────────────
  const handleFile = useCallback(async (file: File) => {
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.pdf') && !lower.endsWith('.md') && !lower.endsWith('.markdown')) {
      toast.error('Only PDF and Markdown files are supported.');
      return;
    }

    setBusy(true);
    try {
      if (lower.endsWith('.pdf')) {
        const fd = new FormData();
        fd.append('file', file);
        const res = await fetch(`${API_BASE}/ingest/chunk-preview`, { method: 'POST', body: fd });
        if (!res.ok) { toast.error(`Preview failed (${res.status})`); return; }
        const payload = await res.json() as ChunkPreviewPayload;
        setChunkPreview({ file, payload });
      } else {
        const jobId = crypto.randomUUID();
        setJobs(prev => [...prev, { id: jobId, fileName: file.name, status: 'running', events: [] }]);
        const fd = new FormData();
        fd.append('file', file);
        fd.append('session_id', jobId);
        const res = await fetch(`${API_BASE}/ingest/stream`, { method: 'POST', body: fd });
        if (!res.ok) {
          setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'error', error: res.statusText } : j));
          return;
        }
        await runSse(jobId, res);
      }
    } finally {
      setBusy(false);
    }
  }, [runSse]);

  // ── Ingest without OCR ──────────────────────────────────────────────────────
  const ingestWithoutOcr = useCallback(async () => {
    if (!chunkPreview) return;
    const { file } = chunkPreview;
    setChunkPreview(null);
    const jobId = crypto.randomUUID();
    setJobs(prev => [...prev, { id: jobId, fileName: file.name, status: 'running', events: [] }]);
    setBusy(true);
    const fd = new FormData();
    fd.append('file', file);
    fd.append('session_id', jobId);
    try {
      const res = await fetch(`${API_BASE}/ingest/stream`, { method: 'POST', body: fd });
      if (!res.ok) {
        setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'error', error: res.statusText } : j));
        return;
      }
      await runSse(jobId, res);
    } finally {
      setBusy(false);
    }
  }, [chunkPreview, runSse]);

  // ── Paddle OCR preview ──────────────────────────────────────────────────────
  const runPaddleOcr = useCallback(async (useLayoutReader: boolean) => {
    if (!chunkPreview) return;
    const { file } = chunkPreview;
    setChunkPreview(null);
    const toastId = toast.loading(`Running PaddleOCR on "${file.name}"…`);
    setBusy(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5 * 60 * 1000);
    try {
      const fd = new FormData();
      fd.append('file', file);
      if (useLayoutReader) fd.append('use_layout_reader', 'true');
      const res = await fetch(`${API_BASE}/ingest/paddle-ocr-preview`, { method: 'POST', body: fd, signal: controller.signal });
      if (!res.ok) { toast.error(`PaddleOCR failed (${res.status})`, { id: toastId }); return; }
      const payload = await res.json() as ParsePreviewPayload;
      toast.dismiss(toastId);
      setParsePreview({ file, payload });
    } catch (e) {
      toast.error(e instanceof Error && e.name === 'AbortError' ? 'PaddleOCR timed out.' : 'PaddleOCR failed', { id: toastId });
    } finally {
      clearTimeout(timeout);
      setBusy(false);
    }
  }, [chunkPreview]);

  // ── Continue from OCR preview ───────────────────────────────────────────────
  const continueFromPreview = useCallback(async () => {
    if (!parsePreview) return;
    const { file, payload } = parsePreview;
    const md = payload.markdown ?? '';
    if (!md.trim()) { toast.error('No markdown to ingest.'); return; }
    setContinuingPreview(true);
    setParsePreview(null);
    const jobId = crypto.randomUUID();
    setJobs(prev => [...prev, { id: jobId, fileName: file.name, status: 'running', events: [] }]);
    const fd = new FormData();
    fd.append('markdown', new Blob([md], { type: 'text/plain;charset=utf-8' }), 'parsed.md');
    fd.append('source_filename', file.name);
    fd.append('session_id', jobId);
    fd.append('ocr_source', 'paddle');
    try {
      const res = await fetch(`${API_BASE}/ingest/continue`, { method: 'POST', body: fd });
      if (!res.ok) {
        setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'error', error: res.statusText } : j));
        return;
      }
      await runSse(jobId, res);
    } finally {
      setContinuingPreview(false);
    }
  }, [parsePreview, runSse]);

  // ── Drag-drop ───────────────────────────────────────────────────────────────
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background text-foreground" style={{ fontFamily: "'Host Grotesk', sans-serif", letterSpacing: '-0.02em' }}>

      {/* Header */}
      <div className="border-b border-border">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>
            <span className="text-muted-foreground/40">|</span>
            <span className="text-sm font-semibold">
              Grounded<span className="text-primary">MD</span>
              <span className="text-muted-foreground font-normal ml-2">Knowledge Base</span>
            </span>
          </div>
          <button
            onClick={() => navigate('/app')}
            className="px-4 py-1.5 rounded-sm bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 transition-opacity"
          >
            Launch Chat →
          </button>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-10 space-y-12">

        {/* Why section */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-4 h-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Why this knowledge base</h2>
          </div>
          <div className="rounded-sm border border-border bg-card p-6">
            <p className="text-foreground/80 leading-relaxed text-sm">
              GroundedMD is built on <span className="text-foreground font-medium">7 peer-reviewed NINDS TBI Common Data Elements (CDE)</span> working group publications —
              the authoritative evidence base published by the National Institute of Neurological Disorders and Stroke.
              Every answer the system generates is grounded in these documents: no answer is produced without a traceable citation
              to a specific chunk and graph node. This ensures clinicians in low-connectivity environments receive
              <span className="text-foreground font-medium"> evidence-based guidance</span>, not hallucinated content.
            </p>
            <div className="mt-4 flex items-center gap-6 text-xs text-muted-foreground">
              <div className="flex items-center gap-1.5"><Database className="w-3.5 h-3.5 text-primary" /> Neo4j knowledge graph</div>
              <div className="flex items-center gap-1.5"><FileText className="w-3.5 h-3.5 text-primary" /> 7 NINDS publications</div>
              <div className="flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-primary" /> Gemma 4 · 100% offline</div>
            </div>
          </div>
        </section>

        {/* Publications list */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <FileText className="w-4 h-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Pre-loaded evidence ({PUBLICATIONS.length} documents)</h2>
          </div>
          <div className="space-y-2">
            {PUBLICATIONS.map((pub) => (
              <div key={pub.file} className="rounded-sm border border-border bg-card px-5 py-4 flex gap-4">
                <CheckCircle2 className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-foreground">{pub.title}</p>
                  <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{pub.description}</p>
                  <p className="text-[10px] text-muted-foreground/50 mt-1 font-mono">{pub.file}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Upload section */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <Upload className="w-4 h-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Add new document</h2>
          </div>

          <div
            ref={dropRef}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`
              rounded-sm border-2 border-dashed p-10 text-center transition-colors cursor-pointer
              ${dragging ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50 hover:bg-card'}
            `}
            onClick={() => {
              const input = document.createElement('input');
              input.type = 'file';
              input.accept = '.pdf,.md,.markdown';
              input.onchange = (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (file) handleFile(file);
              };
              input.click();
            }}
          >
            {busy ? (
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">Processing…</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Upload className="w-6 h-6 text-muted-foreground/50" />
                <p className="text-sm text-foreground/70">Drop a PDF or Markdown file here, or click to browse</p>
                <p className="text-xs text-muted-foreground/50">PDF files show a chunk preview first — you can run PaddleOCR for scanned documents</p>
              </div>
            )}
          </div>
        </section>

        {/* Active ingest jobs */}
        {jobs.length > 0 && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-primary" />
              <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Ingest progress</h2>
            </div>
            <div className="space-y-3">
              {jobs.map(job => <JobCard key={job.id} job={job} />)}
            </div>
          </section>
        )}
      </div>

      {/* Modals */}
      <ChunkPreviewModal
        open={!!chunkPreview}
        file={chunkPreview?.file ?? null}
        payload={chunkPreview?.payload ?? null}
        onOpenChange={(open) => { if (!open) setChunkPreview(null); }}
        onIngestWithoutOcr={ingestWithoutOcr}
        onRunPaddleOcr={runPaddleOcr}
        paddleAvailable={paddleAvailable}
        layoutAnalysisAvailable={layoutAvailable}
      />

      <PreviewErrorBoundary onReset={() => setParsePreview(null)}>
        <ParsePreviewModal
          open={!!parsePreview}
          file={parsePreview?.file ?? null}
          payload={parsePreview?.payload ?? null}
          title="Paddle OCR preview"
          groundingSourceLabel="Boxes generated from Paddle OCR output; colors follow chunk types."
          continuing={continuingPreview}
          onOpenChange={(open) => { if (!open) setParsePreview(null); }}
          onContinue={continueFromPreview}
        />
      </PreviewErrorBoundary>
    </div>
  );
}

// ── Job card ─────────────────────────────────────────────────────────────────
function JobCard({ job }: { job: IngestJob }) {
  const [open, setOpen] = useState(true);

  const icon = job.status === 'running'
    ? <Loader2 className="w-3.5 h-3.5 animate-spin text-primary shrink-0" />
    : job.status === 'error'
    ? <AlertCircle className="w-3.5 h-3.5 text-destructive shrink-0" />
    : <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0" />;

  return (
    <div className="rounded-sm border border-border bg-card overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-sidebar-accent/30 transition-colors"
      >
        {icon}
        <span className="text-sm font-medium truncate flex-1 text-foreground">{job.fileName}</span>
        {job.status === 'complete' && (
          <span className="text-[10px] text-primary font-medium mr-2">Ingested</span>
        )}
        {job.status === 'error' && (
          <span className="text-[10px] text-destructive font-medium mr-2">Failed</span>
        )}
        <ChevronDown className={`w-4 h-4 text-muted-foreground/50 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="border-t border-border px-4 py-3 space-y-1 max-h-64 overflow-y-auto">
          {job.error && (
            <p className="text-xs text-destructive">{job.error}</p>
          )}
          {job.events.map((ev, i) => (
            <div key={i} className="text-[11px] text-muted-foreground flex gap-2">
              <span className="text-primary/70 shrink-0">{ev.event_type}</span>
              <span className="truncate">{ev.task_name ?? ev.message ?? ''}</span>
            </div>
          ))}
          {job.status === 'running' && job.events.length === 0 && (
            <p className="text-xs text-muted-foreground">Starting…</p>
          )}
          {job.status === 'complete' && job.graphPreview && (
            <div className="mt-2 pt-2 border-t border-border text-[11px] text-muted-foreground">
              Graph: {(job.graphPreview as Record<string, unknown>).num_nodes as number ?? '?'} nodes ·{' '}
              {(job.graphPreview as Record<string, unknown>).num_relations as number ?? '?'} relationships added
            </div>
          )}
        </div>
      )}
    </div>
  );
}
