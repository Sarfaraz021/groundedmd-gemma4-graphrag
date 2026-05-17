import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';
import { apiFetch, API_BASE } from '@/lib/api';
import ChatSidebar from '@/components/ChatSidebar';
import ChatArea from '@/components/ChatArea';
import ChunkPreviewModal from '@/components/ChunkPreviewModal';
import ParsePreviewModal from '@/components/ParsePreviewModal';
import PreviewErrorBoundary from '@/components/PreviewErrorBoundary';
import BatchOcrChoiceModal, { type BatchOcrMode } from '@/components/BatchOcrChoiceModal';
import { useChatSessions } from '@/hooks/useChatSessions';
import { readSseJsonLines } from '@/lib/sse';
import type {
  ChatMessage,
  IngestStreamEvent,
  GraphPreview,
  IngestMeta,
  ChunkPreviewPayload,
  ParsePreviewPayload,
  Pipeline,
} from '@/lib/types';


interface Props {
  userName?: string;
}

function normalizeIngestEvent(raw: Record<string, unknown>): IngestStreamEvent {
  return {
    event_type: String(raw.event_type ?? ''),
    task_name: raw.task_name != null ? String(raw.task_name) : undefined,
    message: raw.message != null ? String(raw.message) : undefined,
    payload: raw.payload as Record<string, unknown> | undefined,
    timestamp: raw.timestamp != null ? String(raw.timestamp) : undefined,
    hint: raw.hint as IngestStreamEvent['hint'],
  };
}

const Chat = ({ userName = 'Guest' }: Props) => {
  const {
    sessions,
    activeSession,
    activeSessionId,
    setActiveSessionId,
    createSession,
    deleteSession,
    addMessage,
    addIngestJob,
    appendIngestEvent,
    setIngestJobMeta,
    finishIngestJob,
  } = useChatSessions();

  const [activeIngestFileName, setActiveIngestFileName] = useState<string | null>(null);
  const [batchIngesting, setBatchIngesting] = useState(false);
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [activePipelineId, setActivePipelineId] = useState<string | null>(null);
  const [ingestConfig, setIngestConfig] = useState({
    docling_ocr_available: false,
    chunk_preview: false,
    layout_analysis: false,
  });
  const [parsePreview, setParsePreview] = useState<{
    file: File;
    payload: ParsePreviewPayload;
    sessionId: string;
  } | null>(null);
  const [chunkPreview, setChunkPreview] = useState<{
    file: File;
    payload: ChunkPreviewPayload;
    sessionId: string;
  } | null>(null);
  const [continuingPreview, setContinuingPreview] = useState(false);
  const [pendingBatch, setPendingBatch] = useState<{
    files: File[];
    sessionId: string;
    pipelineId: string | null;
  } | null>(null);

  const fetchPipelines = useCallback(() => {
    fetch(`${API_BASE}/pipelines`)
      .then((r) => r.json())
      .then((x: { pipelines?: Pipeline[] }) => {
        if (Array.isArray(x.pipelines)) setPipelines(x.pipelines);
      })
      .catch((e: unknown) => {
        if (e instanceof Error) console.warn('Failed to fetch pipelines:', e.message);
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/ingest/config`, { signal: controller.signal })
      .then((r) => r.json())
      .then((x: {
        paddle_ocr_preview?: boolean;
        docling_ocr_available?: boolean;
        chunk_preview?: boolean;
        layout_analysis?: boolean;
      }) =>
        setIngestConfig({
          docling_ocr_available: Boolean(x.docling_ocr_available ?? x.paddle_ocr_preview),
          chunk_preview: Boolean(x.chunk_preview),
          layout_analysis: Boolean(x.layout_analysis),
        })
      )
      .catch((e: unknown) => {
        if (e instanceof Error && e.name !== 'AbortError') {
          console.warn('Failed to fetch ingest config:', e.message);
        }
      });
    fetchPipelines();
    return () => controller.abort();
  }, [fetchPipelines]);

  /** Clear attachment-row spinner if the ingest job for that file already finished (e.g. race with finally). */
  useEffect(() => {
    const jobs = activeSession?.ingestJobs ?? [];
    if (!activeIngestFileName) return;
    const stillRunning = jobs.some(
      (j) => j.fileName === activeIngestFileName && j.status === 'running'
    );
    if (!stillRunning) {
      setActiveIngestFileName(null);
    }
  }, [activeSession?.ingestJobs, activeIngestFileName]);

  const handleSendMessage = (sessionId: string, message: ChatMessage) => {
    addMessage(sessionId, message);
  };

  const runIngestSse = useCallback(
    async (sessionId: string, jobId: string, res: Response) => {
      let finished = false;
      const finish = (payload: {
        status: 'complete' | 'error';
        graphPreview?: GraphPreview;
        error?: string;
      }) => {
        if (finished) return;
        finished = true;
        finishIngestJob(sessionId, jobId, payload);
      };
      try {
        await readSseJsonLines(res, (data) => {
          const t = data.type as string;
          if (t === 'meta') {
            const meta: IngestMeta = {
              loader: data.loader as string | undefined,
              chunk_config: data.chunk_config as IngestMeta['chunk_config'],
              embedding_model: data.embedding_model as string | undefined,
              markdown_chars: data.markdown_chars as number | undefined,
              parse_preview_continue: data.parse_preview_continue as boolean | undefined,
            };
            setIngestJobMeta(sessionId, jobId, meta);
          } else if (t === 'event' && data.event && typeof data.event === 'object') {
            appendIngestEvent(sessionId, jobId, normalizeIngestEvent(data.event as Record<string, unknown>));
          } else if (t === 'done') {
            finish({
              status: 'complete',
              graphPreview: data.graph_preview as GraphPreview | undefined,
            });
          } else if (t === 'error') {
            finish({
              status: 'error',
              error: String(data.message ?? 'Ingest failed'),
            });
          }
        });
        if (!finished) {
          finish({ status: 'complete' });
        }
      } catch (e) {
        finish({
          status: 'error',
          error: e instanceof Error ? e.message : 'Network error',
        });
      }
    },
    [appendIngestEvent, setIngestJobMeta, finishIngestJob]
  );

  const startIngest = useCallback(
    async (sessionId: string, file: File) => {
      const lower = file.name.toLowerCase();
      if (!lower.endsWith('.pdf') && !lower.endsWith('.md') && !lower.endsWith('.markdown')) {
        return;
      }

      // Step 1: Chunk preview (no OCR) for PDFs.
      if (lower.endsWith('.pdf')) {
        setActiveIngestFileName(file.name);
        try {
          const fd = new FormData();
          fd.append('file', file);
          const res = await apiFetch(`${API_BASE}/ingest/chunk-preview`, { method: 'POST', body: fd });
          if (!res.ok) {
            const t = await res.text();
            toast.error(t || `Chunk preview failed (${res.status})`);
            return;
          }
          const payload = (await res.json()) as ChunkPreviewPayload;
          setChunkPreview({ file, payload, sessionId });
        } catch (e) {
          toast.error(e instanceof Error ? e.message : 'Chunk preview failed');
        } finally {
          setActiveIngestFileName(null);
        }
        return;
      }

      const jobId = crypto.randomUUID();
      addIngestJob(sessionId, {
        id: jobId,
        fileName: file.name,
        startedAt: new Date().toISOString(),
        status: 'running',
        events: [],
      });
      setActiveIngestFileName(file.name);

      const fd = new FormData();
      fd.append('file', file);
      fd.append('session_id', sessionId);

      try {
        const res = await apiFetch(`${API_BASE}/ingest/stream`, { method: 'POST', body: fd });
        if (!res.ok) {
          const t = await res.text();
          finishIngestJob(sessionId, jobId, { status: 'error', error: t || res.statusText });
          return;
        }
        await runIngestSse(sessionId, jobId, res);
      } catch (e) {
        finishIngestJob(sessionId, jobId, {
          status: 'error',
          error: e instanceof Error ? e.message : 'Network error',
        });
      } finally {
        setActiveIngestFileName(null);
      }
    },
    [addIngestJob, runIngestSse, finishIngestJob]
  );

  const ingestWithoutOcrFromChunkPreview = useCallback(async () => {
    if (!chunkPreview) return;
    const { file, sessionId } = chunkPreview;
    setChunkPreview(null);

    const jobId = crypto.randomUUID();
    addIngestJob(sessionId, {
      id: jobId,
      fileName: file.name,
      startedAt: new Date().toISOString(),
      status: 'running',
      events: [],
    });
    setActiveIngestFileName(file.name);

    const fd = new FormData();
    fd.append('file', file);
    fd.append('session_id', sessionId);

    try {
      const res = await apiFetch(`${API_BASE}/ingest/stream`, { method: 'POST', body: fd });
      if (!res.ok) {
        const t = await res.text();
        finishIngestJob(sessionId, jobId, { status: 'error', error: t || res.statusText });
        return;
      }
      await runIngestSse(sessionId, jobId, res);
    } catch (e) {
      finishIngestJob(sessionId, jobId, {
        status: 'error',
        error: e instanceof Error ? e.message : 'Network error',
      });
    } finally {
      setActiveIngestFileName(null);
    }
  }, [chunkPreview, addIngestJob, runIngestSse, finishIngestJob]);

  const runDoclingOcrPreview = useCallback(async (useLayoutReader: boolean) => {
    if (!chunkPreview) return;
    const { file, sessionId } = chunkPreview;
    setChunkPreview(null);

    const label = useLayoutReader ? 'Docling OCR + LayoutReader' : 'Docling OCR';
    const toastId = toast.loading(`Running ${label} on "${file.name}"…`);
    setActiveIngestFileName(file.name);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10 * 60 * 1000); // 10 min for large PDFs

    try {
      const fd = new FormData();
      fd.append('file', file);
      if (useLayoutReader) fd.append('use_layout_reader', 'true');
      const res = await apiFetch(`${API_BASE}/ingest/docling-ocr-preview`, {
        method: 'POST',
        body: fd,
        signal: controller.signal,
      });
      if (!res.ok) {
        const t = await res.text();
        toast.error(t || `Docling OCR preview failed (${res.status})`, { id: toastId });
        return;
      }
      const payload = (await res.json()) as ParsePreviewPayload;
      toast.dismiss(toastId);
      setParsePreview({ file, payload, sessionId });
    } catch (e) {
      const msg = e instanceof Error
        ? (e.name === 'AbortError' ? `${label} timed out.` : e.message)
        : 'Docling OCR preview failed';
      toast.error(msg, { id: toastId });
    } finally {
      clearTimeout(timeout);
      setActiveIngestFileName(null);
    }
  }, [chunkPreview]);

  const continueFromPreview = useCallback(async () => {
      if (!parsePreview) return;
      const { file, payload, sessionId } = parsePreview;
      const md = payload.markdown ?? '';
      if (!md.trim()) {
        toast.error('No markdown to ingest.');
        return;
      }
      setContinuingPreview(true);
      setParsePreview(null);
      const jobId = crypto.randomUUID();
      addIngestJob(sessionId, {
        id: jobId,
        fileName: file.name,
        startedAt: new Date().toISOString(),
        status: 'running',
        events: [],
      });
      setActiveIngestFileName(file.name);

      const fd = new FormData();
      fd.append('markdown', new Blob([md], { type: 'text/plain;charset=utf-8' }), 'parsed.md');
      fd.append('source_filename', file.name);
      fd.append('session_id', sessionId);
      fd.append('ocr_source', 'docling');

      try {
        const res = await apiFetch(`${API_BASE}/ingest/continue`, { method: 'POST', body: fd });
        if (!res.ok) {
          const t = await res.text();
          finishIngestJob(sessionId, jobId, { status: 'error', error: t || res.statusText });
          return;
        }
        await runIngestSse(sessionId, jobId, res);
      } catch (e) {
        finishIngestJob(sessionId, jobId, {
          status: 'error',
          error: e instanceof Error ? e.message : 'Network error',
        });
      } finally {
        setActiveIngestFileName(null);
        setContinuingPreview(false);
      }
    },
    [parsePreview, addIngestJob, runIngestSse, finishIngestJob]
  );

  const startIngestFromUrl = useCallback(
    async (sessionId: string, url: string) => {
      const jobId = crypto.randomUUID();
      // Derive a display name from the URL path
      const displayName = url.split('/').pop()?.split('?')[0] || url;
      addIngestJob(sessionId, {
        id: jobId,
        fileName: displayName,
        startedAt: new Date().toISOString(),
        status: 'running',
        events: [],
      });
      setActiveIngestFileName(displayName);

      try {
        const res = await apiFetch(`${API_BASE}/ingest/from-url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, session_id: sessionId }),
        });
        if (!res.ok) {
          const t = await res.text();
          finishIngestJob(sessionId, jobId, { status: 'error', error: t || res.statusText });
          return;
        }
        await runIngestSse(sessionId, jobId, res);
      } catch (e) {
        finishIngestJob(sessionId, jobId, {
          status: 'error',
          error: e instanceof Error ? e.message : 'Network error',
        });
      } finally {
        setActiveIngestFileName(null);
      }
    },
    [addIngestJob, runIngestSse, finishIngestJob]
  );

  const startBatchIngest = useCallback(
    async (sessionId: string, filesToIngest: File[], pipelineId: string | null = null, ocrMode: BatchOcrMode = 'none') => {
      if (filesToIngest.length === 0) return;
      setBatchIngesting(true);

      // Create all jobs upfront so the sidebar shows them immediately
      const jobs = filesToIngest.map((file) => ({
        id: crypto.randomUUID(),
        fileName: file.name,
        startedAt: new Date().toISOString(),
        status: 'running' as const,
        events: [],
      }));
      jobs.forEach((job) => addIngestJob(sessionId, job));

      const fd = new FormData();
      filesToIngest.forEach((f) => fd.append('files', f));
      fd.append('session_id', sessionId);
      fd.append('ocr_mode', ocrMode);
      if (pipelineId) fd.append('pipeline_id', pipelineId);

      // Track which jobs have been finished to handle early stream end
      const finished = new Set<number>();

      try {
        const res = await apiFetch(`${API_BASE}/ingest/batch`, { method: 'POST', body: fd });
        if (!res.ok) {
          const t = await res.text();
          jobs.forEach((job) =>
            finishIngestJob(sessionId, job.id, { status: 'error', error: t || res.statusText })
          );
          return;
        }

        await readSseJsonLines(res, (data) => {
          const t = data.type as string;

          // progress event carries queue position — no job routing needed
          if (t === 'progress') return;

          const idx = data.file_index;
          if (typeof idx !== 'number' || !Number.isInteger(idx) || idx < 0 || idx >= jobs.length) return;
          const job = jobs[idx];

          if (t === 'meta') {
            const meta: IngestMeta = {
              loader: data.loader as string | undefined,
              chunk_config: data.chunk_config as IngestMeta['chunk_config'],
              embedding_model: data.embedding_model as string | undefined,
              markdown_chars: data.markdown_chars as number | undefined,
              parse_preview_continue: data.parse_preview_continue as boolean | undefined,
            };
            setIngestJobMeta(sessionId, job.id, meta);
          } else if (t === 'warning') {
            // Retry warning — surface as an ingest event so the sidebar shows it
            appendIngestEvent(sessionId, job.id, normalizeIngestEvent({
              event_type: 'WARNING',
              message: String(data.message ?? ''),
            }));
          } else if (t === 'event' && data.event && typeof data.event === 'object') {
            appendIngestEvent(sessionId, job.id, normalizeIngestEvent(data.event as Record<string, unknown>));
          } else if (t === 'done') {
            finished.add(idx);
            finishIngestJob(sessionId, job.id, {
              status: 'complete',
              graphPreview: data.graph_preview as GraphPreview | undefined,
            });
          } else if (t === 'error') {
            finished.add(idx);
            finishIngestJob(sessionId, job.id, {
              status: 'error',
              error: String(data.message ?? 'Ingest failed'),
            });
          }
        });

        // Finish any jobs the stream didn't explicitly close
        jobs.forEach((job, i) => {
          if (!finished.has(i)) {
            finishIngestJob(sessionId, job.id, { status: 'complete' });
          }
        });
      } catch (e) {
        jobs.forEach((job) =>
          finishIngestJob(sessionId, job.id, {
            status: 'error',
            error: e instanceof Error ? e.message : 'Network error',
          })
        );
      } finally {
        setBatchIngesting(false);
        fetchPipelines();
      }
    },
    [addIngestJob, appendIngestEvent, setIngestJobMeta, finishIngestJob, fetchPipelines]
  );

  return (
    <div className="flex h-screen">
      <BatchOcrChoiceModal
        open={pendingBatch != null}
        fileCount={pendingBatch?.files.length ?? 0}
        pdfCount={pendingBatch?.files.filter(f => f.name.toLowerCase().endsWith('.pdf')).length ?? 0}
        doclingAvailable={ingestConfig.docling_ocr_available}
        onConfirm={(mode) => {
          if (!pendingBatch) return;
          const { files, sessionId, pipelineId } = pendingBatch;
          setPendingBatch(null);
          void startBatchIngest(sessionId, files, pipelineId, mode);
        }}
        onCancel={() => setPendingBatch(null)}
      />

      <PreviewErrorBoundary onReset={() => setParsePreview(null)}>
        <ParsePreviewModal
          open={parsePreview != null}
          file={parsePreview?.file ?? null}
          payload={parsePreview?.payload ?? null}
          title="Docling OCR preview"
          groundingSourceLabel="Boxes generated from Docling OCR output; colors follow chunk types."
          onOpenChange={(open) => {
            if (!open) setParsePreview(null);
          }}
          continuing={continuingPreview}
          onContinue={() => void continueFromPreview()}
        />
      </PreviewErrorBoundary>

      <PreviewErrorBoundary onReset={() => setChunkPreview(null)}>
        <ChunkPreviewModal
          open={chunkPreview != null}
          file={chunkPreview?.file ?? null}
          payload={chunkPreview?.payload ?? null}
          onOpenChange={(open) => {
            if (!open) setChunkPreview(null);
          }}
          onIngestWithoutOcr={() => void ingestWithoutOcrFromChunkPreview()}
          onRunDoclingOcr={(useLayoutReader) => void runDoclingOcrPreview(useLayoutReader)}
          doclingAvailable={ingestConfig.docling_ocr_available}
          layoutAnalysisAvailable={ingestConfig.layout_analysis}
        />
      </PreviewErrorBoundary>

      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewSession={createSession}
        onDeleteSession={deleteSession}
        userName={userName}
        ingestJobs={activeSession?.ingestJobs ?? []}
        pipelines={pipelines}
        activePipelineId={activePipelineId}
        onSelectPipeline={setActivePipelineId}
      />
      <ChatArea
        session={activeSession}
        onSendMessage={handleSendMessage}
        onCreateSession={createSession}
        sessionId={activeSessionId}
        onIngestFile={(file) => {
          const sid = activeSessionId ?? createSession();
          void startIngest(sid, file);
        }}
        onIngestAllFiles={(files, pipelineId) => {
          const sid = activeSessionId ?? createSession();
          setPendingBatch({ files, sessionId: sid, pipelineId });
        }}
        onIngestFromUrl={(url) => {
          const sid = activeSessionId ?? createSession();
          void startIngestFromUrl(sid, url);
        }}
        activeIngestFileName={activeIngestFileName}
        batchIngesting={batchIngesting}
        activePipelineId={activePipelineId}
      />
    </div>
  );
};

export default Chat;
