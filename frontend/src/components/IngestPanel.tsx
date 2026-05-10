import { useState } from 'react';
import {
  ChevronDown,
  CheckCircle2,
  Loader2,
  AlertCircle,
  FileUp,
  Network,
} from 'lucide-react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import type { IngestJob } from '@/lib/types';

interface Props {
  jobs: IngestJob[];
}

export default function IngestPanel({ jobs }: Props) {
  if (jobs.length === 0) return null;

  return (
    <div className="px-2 pb-3 border-b border-sidebar-border">
      <div className="px-2 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground/80">
        Knowledge graph ingest
      </div>
      <div className="space-y-2 max-h-[min(420px,50vh)] overflow-y-auto pr-1">
        {jobs.map((job) => (
          <IngestJobCard key={job.id} job={job} />
        ))}
      </div>
    </div>
  );
}

function IngestJobCard({ job }: { job: IngestJob }) {
  const [open, setOpen] = useState(true);
  const statusIcon =
    job.status === 'running' ? (
      <Loader2 className="w-3.5 h-3.5 animate-spin text-primary shrink-0" />
    ) : job.status === 'error' ? (
      <AlertCircle className="w-3.5 h-3.5 text-destructive shrink-0" />
    ) : (
      <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0" />
    );

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="border border-sidebar-border bg-sidebar-accent/20">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-2 py-2 text-left hover:bg-sidebar-accent/40 transition-colors">
        <FileUp className="w-3 h-3 shrink-0 opacity-50" />
        <span className="text-[11px] font-medium truncate flex-1 text-sidebar-foreground">{job.fileName}</span>
        {statusIcon}
        <ChevronDown className="w-3.5 h-3.5 shrink-0 opacity-40 transition-transform group-data-[state=open]:rotate-180" />
      </CollapsibleTrigger>
      <CollapsibleContent className="border-t border-sidebar-border px-2 py-2 space-y-2 text-[10px] text-muted-foreground">
        {job.meta && (
          <div className="space-y-1 rounded-sm bg-background/40 px-2 py-1.5">
            <div>
              <span className="text-primary/90 font-semibold">Loader · </span>
              {job.meta.loader ?? '—'}
            </div>
            {job.meta.chunk_config && (
              <div>
                <span className="text-primary/90 font-semibold">Chunks · </span>
                size {job.meta.chunk_config.size}, overlap {job.meta.chunk_config.overlap}
              </div>
            )}
            {job.meta.embedding_model && (
              <div>
                <span className="text-primary/90 font-semibold">Embeddings · </span>
                {job.meta.embedding_model}
              </div>
            )}
          </div>
        )}

        <div className="max-h-40 overflow-y-auto space-y-1 font-mono leading-relaxed">
          {job.events.map((ev, i) => {
            const title = ev.hint?.title ?? ev.task_name ?? ev.event_type;
            const detail = ev.hint?.detail;
            if (ev.event_type === 'TASK_PROGRESS' && ev.message) {
              return (
                <div key={i} className="text-[9px] text-muted-foreground/90">
                  … {ev.message}
                </div>
              );
            }
            if (ev.event_type === 'TASK_STARTED') {
              return (
                <div key={i} className="text-[10px]">
                  <span className="text-primary font-semibold">▸ {title}</span>
                  {detail ? <span className="opacity-80"> — {detail}</span> : null}
                </div>
              );
            }
            if (ev.event_type === 'TASK_FINISHED') {
              return (
                <div key={i} className="text-[10px] text-foreground/80">
                  <span className="text-primary/70">✓ {title}</span>
                </div>
              );
            }
            if (ev.event_type === 'PIPELINE_STARTED' || ev.event_type === 'PIPELINE_FINISHED') {
              return (
                <div key={i} className="text-[9px] uppercase tracking-wide opacity-60">
                  {ev.event_type.replace(/_/g, ' ')}
                </div>
              );
            }
            return null;
          })}
        </div>

        {job.error && (
          <p className="text-[10px] text-destructive leading-snug">{job.error}</p>
        )}

        {job.graphPreview && job.status === 'complete' && (
          <div className="rounded-sm border border-border bg-background/50 p-2 space-y-2">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
              <Network className="w-3 h-3" />
              Neo4j snapshot
            </div>
            {job.graphPreview.stats && (
              <div className="grid grid-cols-3 gap-1 text-center text-[10px]">
                <div>
                  <div className="text-muted-foreground">Chunks</div>
                  <div className="font-semibold text-foreground">{job.graphPreview.stats.chunks ?? 0}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Entities</div>
                  <div className="font-semibold text-foreground">{job.graphPreview.stats.entities ?? 0}</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Rels</div>
                  <div className="font-semibold text-foreground">{job.graphPreview.stats.relationships ?? 0}</div>
                </div>
              </div>
            )}
            {job.graphPreview.links && job.graphPreview.links.length > 0 && (
              <ul className="max-h-28 overflow-y-auto space-y-0.5 text-[9px] leading-tight">
                {job.graphPreview.links.slice(0, 24).map((l, i) => (
                  <li key={i} className="truncate opacity-90">
                    <span className="text-foreground/90">{l.source_name}</span>
                    <span className="text-primary mx-0.5">—[{l.type}]→</span>
                    <span className="text-foreground/90">{l.target_name}</span>
                  </li>
                ))}
                {job.graphPreview.links.length > 24 && (
                  <li className="opacity-50">+{job.graphPreview.links.length - 24} more…</li>
                )}
              </ul>
            )}
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
