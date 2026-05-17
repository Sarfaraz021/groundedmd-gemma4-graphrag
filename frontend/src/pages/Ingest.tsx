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
    title: 'AI for TBI Imaging: Translational Review',
    file: 'ai-for-tbi-imaging-translational-review-furst-2026.pdf',
    description: 'Comprehensive review of AI algorithms for TBI imaging — from model development to clinical implementation — covering CT, MRI, and deep learning diagnostic tools. Frontiers in Neurology, 2026.',
  },
  {
    title: 'Conformal Prediction for ICH Detection',
    file: 'conformal-prediction-ich-detection-gamble-2024.pdf',
    description: 'Applies conformal prediction to a deep learning model for intracranial hemorrhage detection, improving trustworthiness through statistically rigorous uncertainty quantification. Radiology: AI, 2024.',
  },
  {
    title: 'Bridging the Trust Gap: Conformal Prediction for ICH',
    file: 'trust-gap-conformal-prediction-ich-ngum-filippi-2025.pdf',
    description: 'Commentary on integrating conformal prediction into AI-based intracranial hemorrhage detection workflows to close the clinical trust gap. Radiology: AI, 2025.',
  },
  {
    title: 'Blood Biomarkers for ICH and Outcome in Moderate-Severe TBI',
    file: 'blood-biomarkers-ich-outcome-moderate-severe-tbi-anderson.pdf',
    description: 'Investigates blood-based biomarkers — including GFAP, UCH-L1, and S100B — for predicting intracranial hemorrhage and functional outcome in moderate-to-severe TBI patients.',
  },
  {
    title: 'Outcome Prediction in Severe TBI Using Deep Learning from CT',
    file: 'outcome-prediction-severe-tbi-deep-learning-ct.pdf',
    description: 'Deep learning model trained on head CT scans to predict functional outcomes in patients with severe traumatic brain injury — validated on a multi-centre cohort.',
  },
  {
    title: 'Data-Driven Prognosis in TBI with Interpretable ML',
    file: 'data-driven-prognosis-tbi-interpretable-ml-tritt-2023.pdf',
    description: 'Combines data-driven feature distillation with interpretable machine learning to improve precision prognosis in TBI, providing clinically explainable outcome predictions. Scientific Reports, 2023.',
  },
  {
    title: 'Refining TBI Outcome Prediction with Machine Learning',
    file: 'outcome-prediction-tbi-machine-learning-bark-2024.pdf',
    description: 'Benchmarks and refines multiple machine learning algorithms for predicting long-term outcomes after traumatic brain injury using structured clinical and imaging variables. Scientific Reports, 2024.',
  },
  {
    title: 'TBI and AI: Shaping the Future of Neurorehabilitation',
    file: 'tbi-and-ai-neurorehabilitation-review-orenuga-2025.pdf',
    description: 'Review of how artificial intelligence is transforming TBI neurorehabilitation — from personalised recovery plans to adaptive therapy tools and outcome monitoring. Life, 2025.',
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
  const [doclingAvailable, setDoclingAvailable] = useState(false);
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
        setDoclingAvailable(Boolean(x.docling_ocr_available ?? x.paddle_ocr_preview));
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

  // ── Docling OCR preview ─────────────────────────────────────────────────────
  const runDoclingOcr = useCallback(async (_useLayoutReader: boolean) => {
    if (!chunkPreview) return;
    const { file } = chunkPreview;
    setChunkPreview(null);
    const toastId = toast.loading(`Running Docling OCR on "${file.name}"…`);
    setBusy(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10 * 60 * 1000);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_BASE}/ingest/docling-ocr-preview`, { method: 'POST', body: fd, signal: controller.signal });
      if (!res.ok) { toast.error(`Docling OCR failed (${res.status})`, { id: toastId }); return; }
      const payload = await res.json() as ParsePreviewPayload;
      toast.dismiss(toastId);
      setParsePreview({ file, payload });
    } catch (e) {
      toast.error(e instanceof Error && e.name === 'AbortError' ? 'Docling OCR timed out.' : 'Docling OCR failed', { id: toastId });
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
    fd.append('ocr_source', 'docling');
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
              GroundedMD is built on <span className="text-foreground font-medium">evidence-based clinical TBI research papers</span> covering
              AI diagnostics, blood biomarkers, outcome prediction, and neurorehabilitation — spanning Frontiers in Neurology, Radiology: AI, Scientific Reports, and Life.
              Every answer the system generates is grounded in these documents: no answer is produced without a traceable citation
              to a specific chunk and graph node. This ensures clinicians in low-connectivity environments receive
              <span className="text-foreground font-medium"> evidence-based guidance</span>, not hallucinated content. The corpus can be
              extended with additional peer-reviewed papers to broaden coverage.
            </p>
            <div className="mt-4 flex items-center gap-6 text-xs text-muted-foreground">
              <div className="flex items-center gap-1.5"><Database className="w-3.5 h-3.5 text-primary" /> Neo4j knowledge graph</div>
              <div className="flex items-center gap-1.5"><FileText className="w-3.5 h-3.5 text-primary" /> Peer-reviewed TBI publications</div>
              <div className="flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-primary" /> Gemma 4 · 100% offline</div>
            </div>
          </div>
        </section>

        {/* Publications list */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <FileText className="w-4 h-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">Sample evidence corpus</h2>
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
                <p className="text-xs text-muted-foreground/50">PDF files show a chunk preview first — you can run Docling OCR for advanced document understanding</p>
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
        onRunDoclingOcr={runDoclingOcr}
        doclingAvailable={doclingAvailable}
        layoutAnalysisAvailable={layoutAvailable}
      />

      <PreviewErrorBoundary onReset={() => setParsePreview(null)}>
        <ParsePreviewModal
          open={!!parsePreview}
          file={parsePreview?.file ?? null}
          payload={parsePreview?.payload ?? null}
          title="Docling OCR preview"
          groundingSourceLabel="Boxes generated from Docling OCR output; colors follow chunk types."
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
