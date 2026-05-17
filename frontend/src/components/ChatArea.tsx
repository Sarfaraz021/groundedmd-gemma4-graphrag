import { useState, useRef, useEffect, useCallback } from 'react';
import type { EvalScores, SourceChunk } from '@/lib/types';
import {
  Send,
  Paperclip,
  FileText,
  Link,
  X,
  Bot,
  User,
  Copy,
  Check,
  ChevronDown,
  Sparkles,
  Loader2,
  GitBranch,
  Database,
  PackageOpen,
} from 'lucide-react';
import { AssistantMarkdown } from '@/components/AssistantMarkdown';
import EvalScoresCard from '@/components/EvalScoresCard';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { apiFetch, API_BASE } from '@/lib/api';
import type { ChatSession, ChatMessage, PipelineStep } from '@/lib/types';

function parseSourceChunks(raw: unknown): SourceChunk[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (c): c is SourceChunk =>
      c !== null &&
      typeof c === 'object' &&
      typeof (c as Record<string, unknown>).index === 'number' &&
      typeof (c as Record<string, unknown>).text === 'string',
  );
}

function isIngestable(name: string): boolean {
  const n = name.toLowerCase();
  return n.endsWith('.pdf') || n.endsWith('.md') || n.endsWith('.markdown');
}


interface Props {
  session: ChatSession | null;
  onSendMessage: (sessionId: string, message: ChatMessage) => void;
  onCreateSession: () => string;
  /** When set, PDF/MD chips show “Ingest to Neo4j” for streaming pipeline progress in the sidebar. */
  sessionId: string | null;
  onIngestFile?: (file: File) => void;
  onIngestAllFiles?: (files: File[], pipelineId: string | null) => void;
  onIngestFromUrl?: (url: string) => void;
  activeIngestFileName?: string | null;
  batchIngesting?: boolean;
  /** Active pipeline ID for retrieval filtering (passed to chat/stream). */
  activePipelineId?: string | null;
}

type Phase = 'idle' | 'retrieving' | 'streaming';

export default function ChatArea({
  session,
  onSendMessage,
  onCreateSession,
  sessionId,
  onIngestFile,
  onIngestAllFiles,
  onIngestFromUrl,
  activeIngestFileName,
  batchIngesting,
  activePipelineId,
}: Props) {
  const [input, setInput] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [batchPipelineId, setBatchPipelineId] = useState('');
  const [urlInputOpen, setUrlInputOpen] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [streamingText, setStreamingText] = useState('');
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [streamError, setStreamError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const busy = phase !== 'idle';

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [session?.messages, streamingText, phase, pipelineSteps, scrollToBottom]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (streamTimerRef.current) clearTimeout(streamTimerRef.current);
    };
  }, []);

  const autoResize = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  };

  const pendingEvalRef = useRef<EvalScores | null>(null);

  /** Reveal answer word-by-word for a smooth, readable stream. */
  const streamAnswerText = useCallback(
    (fullText: string, sessionId: string, sourceChunks: SourceChunk[] | undefined, retrievedContext?: string) => {
      const parts = fullText.split(/(\s+)/);
      let i = 0;
      let acc = '';

      const step = () => {
        if (i >= parts.length) {
          streamTimerRef.current = null;
          setPhase('idle');
          setStreamingText('');
          setPipelineSteps([]);
          const msg: ChatMessage = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: fullText || 'No answer returned.',
            timestamp: new Date(),
            sourceChunks,
            retrievedContext,
            evalScores: pendingEvalRef.current ?? undefined,
          };
          pendingEvalRef.current = null;
          onSendMessage(sessionId, msg);
          return;
        }
        acc += parts[i];
        setStreamingText(acc);
        i += 1;
        const piece = parts[i - 1] ?? '';
        const delay = piece.length > 60 ? 12 : piece === '\n' ? 28 : 20;
        streamTimerRef.current = window.setTimeout(step, delay);
      };

      streamTimerRef.current = window.setTimeout(step, 45);
    },
    [onSendMessage]
  );

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed && files.length === 0) return;
    if (busy) return;

    let sid = session?.id;
    if (!sid) sid = onCreateSession();

    const content =
      files.length > 0 ? `${trimmed}\n\n📎 ${files.map((f) => f.name).join(', ')}` : trimmed;
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    onSendMessage(sid!, userMsg);
    setInput('');
    setFiles([]);
    setStreamError(null);
    setPipelineSteps([]);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    if (streamTimerRef.current) {
      clearTimeout(streamTimerRef.current);
      streamTimerRef.current = null;
    }
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const signal = abortRef.current.signal;

    setPhase('retrieving');
    setStreamingText('');

    try {
      console.log('[ChatArea] fetching', `${API_BASE}/chat/stream`);
      const res = await apiFetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ message: trimmed, top_k: 15, pipeline_id: activePipelineId ?? null }),
        signal,
      });
      console.log('[ChatArea] response status:', res.status);

      if (!res.ok) {
        const errText = await res.text();
        console.error('[ChatArea] non-ok response:', errText);
        throw new Error(errText || `Request failed (${res.status})`);
      }

      if (!res.body) {
        throw new Error('No response body from /chat/stream');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let resultReceived = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() ?? '';
        for (const block of blocks) {
          for (const line of block.split('\n')) {
            if (!line.startsWith('data:')) continue;
            const raw = line.slice(5).trim();
            if (!raw) continue;
            let evt: Record<string, unknown>;
            try {
              evt = JSON.parse(raw) as Record<string, unknown>;
            } catch {
              continue;
            }
            const t = evt.type;
            if (t === 'step') {
              setPipelineSteps((prev) => [
                ...prev,
                {
                  phase: String(evt.phase ?? ''),
                  title: String(evt.title ?? ''),
                  detail: String(evt.detail ?? ''),
                },
              ]);
            } else if (t === 'result') {
              const answer = String(evt.answer ?? '').trim();
              const chunks = parseSourceChunks(evt.source_chunks);
              const retrievedContext = typeof evt.context === 'string' ? evt.context : undefined;
              if (signal.aborted) return;
              resultReceived = true;
              setPhase('streaming');
              setStreamingText('');
              streamAnswerText(answer, sid!, chunks, retrievedContext);
              // Don't return — keep reading for the evaluation event
            } else if (t === 'evaluation') {
              pendingEvalRef.current = {
                faithfulness:    Number(evt.faithfulness    ?? -1),
                completeness:    Number(evt.completeness    ?? -1),
                relevance:       Number(evt.relevance       ?? -1),
                context_quality: Number(evt.context_quality ?? -1),
                overall:         Number(evt.overall         ?? -1),
                reasoning:       String(evt.reasoning       ?? ''),
              };
            } else if (t === 'error') {
              throw new Error(String(evt.message ?? 'Stream error'));
            }
          }
        }
      }

      if (signal.aborted) return;
      if (!resultReceived) throw new Error('Stream ended without a result event');
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      const msg = e instanceof Error ? e.message : 'Network error';
      console.error('[ChatArea] stream error:', e);
      setStreamError(msg);
      setPhase('idle');
      setStreamingText('');
      setPipelineSteps([]);
      onSendMessage(sid!, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Could not reach the GraphRAG API (${API_BASE}).\n\n${msg}\n\nStart the backend: uvicorn api.main:app --port 8000`,
        timestamp: new Date(),
      });
    }
  };

  const submitUrlIngest = () => {
    const url = urlInput.trim();
    if (!url || !onIngestFromUrl || !sessionId) return;
    onIngestFromUrl(url);
    setUrlInput('');
    setUrlInputOpen(false);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
  };
  const removeFile = (i: number) => setFiles((prev) => prev.filter((_, idx) => idx !== i));
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const messages = session?.messages ?? [];
  const showAssistantStream = phase === 'retrieving' || phase === 'streaming';

  return (
    <div className="flex-1 flex flex-col h-screen bg-background overflow-hidden">
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {messages.length === 0 && !showAssistantStream ? (
          <div className="h-full flex items-center justify-center px-6">
            <div className="text-center space-y-5 max-w-md animate-fade-in">
              <div className="w-14 h-14 mx-auto bg-card border border-border flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-primary" />
              </div>
              <h2 className="text-lg font-semibold text-foreground tracking-tight">TBI evidence assistant</h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Ask about blood-based biomarkers, AI diagnostics, outcome prediction, or neurorehabilitation. Answers are grounded in
                evidence-based clinical TBI research papers with traceable citations.
              </p>
              <div className="flex justify-center gap-6 pt-2">
                {['GRAPH', 'EVIDENCE', 'CITATIONS'].map((t) => (
                  <span
                    key={t}
                    className="text-[10px] font-semibold uppercase tracking-[0.15em] text-primary/60"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="pb-6">
            {messages.map((msg) => (
              <MessageRow key={msg.id} message={msg} />
            ))}
            {showAssistantStream && (
              <div className="py-6 bg-card/30 border-t border-border/40">
                <div className="max-w-[700px] mx-auto px-6 flex gap-4">
                  <div className="w-7 h-7 shrink-0 bg-primary/10 border border-primary/20 flex items-center justify-center mt-1">
                    <Bot className="w-3.5 h-3.5 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1 space-y-3">
                    <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground/60 block">
                      Assistant
                    </span>
                    {(phase === 'retrieving' || phase === 'streaming') && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-primary/80">
                          <GitBranch className="w-3.5 h-3.5 shrink-0" />
                          Retrieval pipeline
                        </div>
                        {pipelineSteps.length === 0 && phase === 'retrieving' && (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="w-4 h-4 animate-spin text-primary shrink-0" />
                            <span>Starting embedding and graph retrieval…</span>
                          </div>
                        )}
                        {pipelineSteps.length > 0 && (
                          <ol className="space-y-3 border-l border-border pl-3 ml-1">
                            {pipelineSteps.map((s, idx) => (
                              <li key={`${s.phase}-${idx}`} className="relative">
                                <span className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-primary ring-2 ring-background" />
                                <div className="text-[13px] font-medium text-foreground leading-snug">{s.title}</div>
                                <p className="text-[11px] text-muted-foreground leading-relaxed mt-0.5">{s.detail}</p>
                              </li>
                            ))}
                          </ol>
                        )}
                        {phase === 'retrieving' && pipelineSteps.length > 0 && (
                          <div className="flex items-center gap-2 text-xs text-muted-foreground pt-1">
                            <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                            <span>
                              {pipelineSteps[pipelineSteps.length - 1]?.phase === 'llm'
                                ? 'Generating grounded answer…'
                                : 'Running next stage…'}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                    {phase === 'streaming' && (
                      <div className="min-h-[1.75rem] break-words" style={{ overflowWrap: 'anywhere' }}>
                        <AssistantMarkdown content={streamingText} />
                        <span className="inline-block w-[2px] h-[1.1em] ml-0.5 bg-primary align-[-0.15em] animate-pulse shadow-[0_0_8px_hsl(var(--primary))]" />
                      </div>
                    )}
                    {streamError && (
                      <p className="text-sm text-destructive">{streamError}</p>
                    )}
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {files.length > 0 && (
        <div className="shrink-0 max-w-[700px] mx-auto w-full px-6 pb-2 space-y-2">
          {(() => {
            const ingestableFiles = files.filter(f => isIngestable(f.name));
            const isSingle = ingestableFiles.length === 1;
            const showSingleBtn = onIngestFile && isSingle;
            const showBatchBtn = onIngestAllFiles && ingestableFiles.length >= 2;
            if (showSingleBtn) {
              const singleFile = ingestableFiles[0];
              return (
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onIngestFile(singleFile)}
                    disabled={busy || batchIngesting || activeIngestFileName === singleFile.name}
                    className="flex items-center gap-1.5 px-3 py-1 bg-primary text-primary-foreground text-[11px] font-semibold uppercase tracking-[0.1em] hover:opacity-90 disabled:opacity-30 transition-all shrink-0"
                    title="Ingest document — shows chunk/OCR preview before ingesting"
                  >
                    {activeIngestFileName === singleFile.name ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Database className="w-3 h-3" />
                    )}
                    {activeIngestFileName === singleFile.name ? 'Preparing…' : 'Ingest'}
                  </button>
                </div>
              );
            }
            if (showBatchBtn) {
              return (
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      name="batch-pipeline-id"
                      value={batchPipelineId}
                      onChange={(e) => setBatchPipelineId(e.target.value)}
                      placeholder="Pipeline ID (e.g. no_ocr, docling_ocr)"
                      className="flex-1 min-w-0 px-2 py-1 bg-card border border-border text-[11px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50"
                    />
                    <button
                      type="button"
                      onClick={() => onIngestAllFiles(ingestableFiles, batchPipelineId.trim() || null)}
                      disabled={busy || batchIngesting || !!activeIngestFileName}
                      className="flex items-center gap-1.5 px-3 py-1 bg-primary text-primary-foreground text-[11px] font-semibold uppercase tracking-[0.1em] hover:opacity-90 disabled:opacity-30 transition-all shrink-0"
                      title="Ingest all documents to Neo4j"
                    >
                      {batchIngesting ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <PackageOpen className="w-3 h-3" />
                      )}
                      {batchIngesting ? 'Ingesting…' : `Ingest all (${ingestableFiles.length})`}
                    </button>
                  </div>
                </div>
              );
            }
            return null;
          })()}
          <div className="flex gap-2 flex-wrap">
            {files.map((file, i) => (
              <div
                key={i}
                className="flex items-center gap-2 px-3 py-1.5 bg-card border border-border text-xs text-muted-foreground"
              >
                <FileText className="w-3 h-3 shrink-0 text-primary" />
                <span className="truncate max-w-[140px]">{file.name}</span>
                {onIngestFile && isIngestable(file.name) && (
                  <button
                    type="button"
                    onClick={() => onIngestFile(file)}
                    disabled={busy || batchIngesting || activeIngestFileName === file.name}
                    className="shrink-0 p-0.5 text-primary hover:opacity-80 disabled:opacity-30"
                    title="Ingest this file to Neo4j (with OCR preview)"
                  >
                    {activeIngestFileName === file.name ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Database className="w-3.5 h-3.5" />
                    )}
                  </button>
                )}
                <button type="button" onClick={() => removeFile(i)} className="hover:text-destructive shrink-0">
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="shrink-0 pb-5 pt-2 px-4">
        <div className="max-w-[700px] mx-auto">
          {urlInputOpen && onIngestFromUrl && sessionId && (
            <div className="flex items-center gap-2 mb-2 bg-card border border-border px-3 py-2">
              <Link className="w-3.5 h-3.5 shrink-0 text-primary" />
              <input
                autoFocus
                type="url"
                name="ingest-url"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { e.preventDefault(); submitUrlIngest(); }
                  if (e.key === 'Escape') { setUrlInputOpen(false); setUrlInput(''); }
                }}
                placeholder="https://example.com/document.pdf"
                className="flex-1 min-w-0 bg-transparent text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none"
              />
              <button
                type="button"
                onClick={submitUrlIngest}
                disabled={!urlInput.trim()}
                className="text-[11px] font-semibold uppercase tracking-wide text-primary hover:opacity-80 disabled:opacity-30 shrink-0"
              >
                Ingest
              </button>
              <button
                type="button"
                onClick={() => { setUrlInputOpen(false); setUrlInput(''); }}
                className="text-muted-foreground hover:text-foreground shrink-0"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
          <div className="flex items-end gap-2 bg-card border border-border p-2">
            <input
              ref={fileInputRef}
              type="file"
              name="documents"
              multiple
              accept=".pdf,.md,.markdown,.csv,.txt,.doc,.docx,.xls,.xlsx,.json"
              onChange={handleFileSelect}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-muted-foreground hover:text-primary transition-colors shrink-0"
              title="Upload documents"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            {onIngestFromUrl && sessionId && (
              <button
                type="button"
                onClick={() => setUrlInputOpen((v) => !v)}
                className={`p-2 transition-colors shrink-0 ${urlInputOpen ? 'text-primary' : 'text-muted-foreground hover:text-primary'}`}
                title="Ingest from URL"
              >
                <Link className="w-4 h-4" />
              </button>
            )}
            <textarea
              ref={textareaRef}
              name="chat-message"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                autoResize();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask about TBI evidence…"
              rows={1}
              disabled={busy}
              className="flex-1 min-w-0 py-2 px-2 bg-transparent text-foreground text-sm placeholder:text-muted-foreground/50 focus:outline-none resize-none leading-relaxed disabled:opacity-50"
              style={{ height: 'auto', maxHeight: '160px' }}
            />
            <button
              type="button"
              onClick={() => void handleSend()}
              disabled={busy || (!input.trim() && files.length === 0)}
              className="p-2 bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-30 transition-all shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-[10px] text-muted-foreground/40 text-center mt-2 tracking-wide uppercase">
            GraphRAG · {API_BASE.replace(/^https?:\/\//, '')}
          </p>
        </div>
      </div>
    </div>
  );
}

function MessageRow({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    void navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const chunks = message.sourceChunks ?? [];
  const retrievedContext = message.retrievedContext;

  return (
    <div className={`group py-6 ${isUser ? '' : 'bg-card/30'}`}>
      <div className="max-w-[700px] mx-auto px-6 flex gap-4">
        <div
          className={`w-7 h-7 shrink-0 flex items-center justify-center mt-1 ${
            isUser ? 'bg-primary text-primary-foreground' : 'bg-primary/10 border border-primary/20'
          }`}
        >
          {isUser ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5 text-primary" />}
        </div>
        <div className="min-w-0 flex-1">
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground/60 mb-1.5 block">
            {isUser ? 'You' : 'Assistant'}
          </span>
          {isUser ? (
            <div
              className="text-[14px] leading-7 text-foreground break-words whitespace-pre-wrap"
              style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}
            >
              {message.content}
            </div>
          ) : (
            <div className="break-words" style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}>
              <AssistantMarkdown content={message.content} />
              {message.evalScores && <EvalScoresCard scores={message.evalScores} />}
            </div>
          )}
          {!isUser && chunks.length > 0 && (
            <Collapsible className="mt-4 border border-border bg-background/50">
              <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground hover:bg-muted/30 transition-colors">
                <span>Source chunks ({chunks.length})</span>
                <ChevronDown className="w-4 h-4 shrink-0 transition-transform group-data-[state=open]:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent className="border-t border-border">
                <ul className="max-h-[420px] overflow-y-auto divide-y divide-border">
                  {chunks.map((c) => (
                    <li key={c.index} className="px-3 py-3 text-xs space-y-1">
                      <div className="flex flex-wrap items-baseline gap-2 text-[10px] uppercase tracking-wide text-primary font-semibold">
                        <span>[{c.index}]</span>
                        <span className="text-foreground font-medium normal-case tracking-normal">{c.document}</span>
                        {c.chunk_index != null && (
                          <span className="text-muted-foreground font-normal normal-case tracking-normal">
                            chunk #{c.chunk_index}
                          </span>
                        )}
                        {c.chunk_node_id && (
                          <span className="text-muted-foreground/70 font-mono font-normal normal-case tracking-normal text-[9px] truncate max-w-[180px]" title={c.chunk_node_id}>
                            {c.chunk_node_id}
                          </span>
                        )}
                      </div>
                      <p className="text-muted-foreground leading-relaxed whitespace-pre-wrap break-words">{c.text}</p>
                    </li>
                  ))}
                </ul>
              </CollapsibleContent>
            </Collapsible>
          )}
          {!isUser && retrievedContext && (
            <Collapsible className="mt-2 border border-border bg-background/50">
              <CollapsibleTrigger className="group flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.1em] text-muted-foreground hover:bg-muted/30 transition-colors">
                <span>LLM context ({retrievedContext.length.toLocaleString()} chars)</span>
                <ChevronDown className="w-4 h-4 shrink-0 transition-transform group-data-[state=open]:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent className="border-t border-border">
                <pre className="max-h-[500px] overflow-y-auto px-3 py-3 text-[10px] leading-relaxed text-muted-foreground whitespace-pre-wrap break-words font-mono">
                  {retrievedContext}
                </pre>
              </CollapsibleContent>
            </Collapsible>
          )}
          {!isUser && (
            <div className="mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                type="button"
                onClick={handleCopy}
                className="p-1 text-muted-foreground hover:text-primary transition-colors"
                title="Copy"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-primary" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
