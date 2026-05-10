import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, API_BASE } from '@/lib/api';
import ForceGraph2D from 'react-force-graph-2d';
import { Loader2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { KnowledgeBaseDocument, KnowledgeBaseGraphPayload } from '@/lib/types';


interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  document: KnowledgeBaseDocument | null;
}

export default function DocumentGraphDialog({ open, onOpenChange, document }: Props) {
  const [data, setData] = useState<KnowledgeBaseGraphPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 640, h: 420 });

  const load = useCallback(async () => {
    if (!document) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const q = new URLSearchParams({ document_id: document.id });
      const res = await apiFetch(`${API_BASE}/knowledge-base/documents/graph?${q.toString()}`);
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const json = (await res.json()) as KnowledgeBaseGraphPayload;
      setData(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load graph');
    } finally {
      setLoading(false);
    }
  }, [document]);

  useEffect(() => {
    if (open && document) void load();
    if (!open) {
      setData(null);
      setError(null);
    }
  }, [open, document, load]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el || !open) return;
    const ro = new ResizeObserver(() => {
      setDims({ w: Math.max(320, el.clientWidth), h: Math.max(320, el.clientHeight) });
    });
    ro.observe(el);
    setDims({ w: Math.max(320, el.clientWidth), h: Math.max(320, el.clientHeight) });
    return () => ro.disconnect();
  }, [open]);

  const graphData = data
    ? {
        nodes: data.nodes.map((n) => ({ ...n })),
        links: data.links.map((l) => ({ ...l })),
      }
    : { nodes: [], links: [] };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[940px] w-[95vw] sm:max-w-[940px] p-0 gap-0 overflow-hidden flex flex-col max-h-[90vh]">
        <DialogHeader className="px-6 pt-6 pb-2 shrink-0">
          <DialogTitle>Graph: {document?.name ?? 'Document'}</DialogTitle>
          <DialogDescription>
            Neo4j subgraph for this document (chunks, extracted entities, and relationships). Chunk-to-chunk{' '}
            <code className="text-[10px]">NEXT_CHUNK</code> links are hidden for readability.
          </DialogDescription>
        </DialogHeader>
        <div className="px-6 pb-2 text-[11px] text-muted-foreground border-b border-border">
          This differs from the neo4j-graphrag <code className="text-[10px]">Pipeline.draw()</code> diagram, which
          visualizes the ingest pipeline DAG, not per-document data.
        </div>
        <div ref={wrapRef} className="relative w-full min-h-[420px] flex-1 bg-muted/20 border-t border-border">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-muted-foreground z-10 bg-background/80">
              <Loader2 className="w-5 h-5 animate-spin" />
              Loading graph…
            </div>
          )}
          {error && !loading && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-destructive px-6">
              {error}
            </div>
          )}
          {!loading && !error && data && graphData.nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-muted-foreground">
              No graph data for this document.
            </div>
          )}
          {!loading && !error && data && graphData.nodes.length > 0 && (
            <ForceGraph2D
              width={dims.w}
              height={dims.h}
              graphData={graphData}
              nodeLabel="name"
              nodeAutoColorBy="group"
              linkLabel={(l: { rel_type?: string }) => l.rel_type ?? ''}
              linkDirectionalArrowLength={3}
              linkDirectionalArrowRelPos={1}
              cooldownTicks={120}
            />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
