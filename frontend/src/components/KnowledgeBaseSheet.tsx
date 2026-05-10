import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { Database, Loader2, Network, Trash2 } from 'lucide-react';
import { toast } from 'sonner';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { apiFetch, API_BASE } from '@/lib/api';
import type { KnowledgeBaseDocument } from '@/lib/types';

const DocumentGraphDialog = lazy(() => import('@/components/DocumentGraphDialog'));


interface Props {
  activePipelineId?: string | null;
}

export default function KnowledgeBaseSheet({ activePipelineId }: Props) {
  const [open, setOpen] = useState(false);
  const [docs, setDocs] = useState<KnowledgeBaseDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<KnowledgeBaseDocument | null>(null);
  const [visualizeDoc, setVisualizeDoc] = useState<KnowledgeBaseDocument | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const url = activePipelineId
        ? `${API_BASE}/knowledge-base/documents?pipeline_id=${encodeURIComponent(activePipelineId)}`
        : `${API_BASE}/knowledge-base/documents`;
      const res = await apiFetch(url);
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = (await res.json()) as { documents: KnowledgeBaseDocument[] };
      setDocs(data.documents ?? []);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to load documents');
      setDocs([]);
    } finally {
      setLoading(false);
    }
  }, [activePipelineId]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  const handleDelete = async () => {
    const d = confirmDelete;
    if (!d) return;
    setConfirmDelete(null);
    setDeletingId(d.id);
    try {
      const q = new URLSearchParams({ document_id: d.id });
      const res = await apiFetch(`${API_BASE}/knowledge-base/documents?${q.toString()}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || res.statusText);
      }
      toast.success(`Removed “${d.name}” from the knowledge base`);
      await load();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Delete failed');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <>
      <Suspense fallback={null}>
        <DocumentGraphDialog
          open={visualizeDoc != null}
          onOpenChange={(o) => {
            if (!o) setVisualizeDoc(null);
          }}
          document={visualizeDoc}
        />
      </Suspense>
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger asChild>
          <button
            type="button"
            className="w-full flex items-center gap-2 px-4 py-2.5 border border-sidebar-border text-[11px] font-semibold uppercase tracking-[0.12em] text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
          >
            <Database className="w-4 h-4 text-primary shrink-0" />
            View Knowledge Base
          </button>
        </SheetTrigger>
        <SheetContent side="right" className="w-full sm:max-w-md flex flex-col gap-0 p-0">
          <SheetHeader className="p-6 pb-2 shrink-0 border-b border-border">
            <SheetTitle className="text-left">Knowledge base</SheetTitle>
            <SheetDescription className="text-left">
              {activePipelineId
                ? `Showing pipeline: ${activePipelineId}. Deleting removes chunks, entities, and the document record.`
                : 'All documents ingested into Neo4j. Deleting removes chunks, unlinked extracted entities, and the document record.'}
            </SheetDescription>
          </SheetHeader>
          <ScrollArea className="flex-1 min-h-0 px-6 py-4">
            {loading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground py-8">
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading…
              </div>
            ) : docs.length === 0 ? (
              <p className="text-sm text-muted-foreground py-6">No documents ingested yet.</p>
            ) : (
              <ul className="space-y-2">
                {docs.map((d) => (
                  <li
                    key={d.id}
                    className="flex items-start gap-3 p-3 border border-border bg-card/50 text-sm"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="font-medium text-foreground break-words leading-snug">{d.name}</p>
                      <p className="text-[11px] text-muted-foreground mt-1">
                        {d.chunk_count} chunk{d.chunk_count === 1 ? '' : 's'}
                      </p>
                    </div>
                    <div className="flex items-center gap-0.5 shrink-0">
                      <button
                        type="button"
                        title="Visualize Neo4j subgraph"
                        onClick={() => setVisualizeDoc(d)}
                        className="p-1.5 text-muted-foreground hover:text-primary hover:bg-primary/10 transition-colors"
                      >
                        <Network className="w-4 h-4" />
                      </button>
                      <button
                        type="button"
                        title="Delete from database"
                        disabled={deletingId === d.id}
                        onClick={() => setConfirmDelete(d)}
                        className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors disabled:opacity-40"
                      >
                        {deletingId === d.id ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Trash2 className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </ScrollArea>
          <div className="p-4 border-t border-border shrink-0">
            <Button type="button" variant="outline" className="w-full" onClick={() => void load()} disabled={loading}>
              Refresh
            </Button>
          </div>
        </SheetContent>
      </Sheet>

      <AlertDialog open={confirmDelete != null} onOpenChange={(o) => !o && setConfirmDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete document?</AlertDialogTitle>
            <AlertDialogDescription>
              This removes all chunks and extracted graph nodes that only belong to “{confirmDelete?.name}”. This cannot
              be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => void handleDelete()}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
