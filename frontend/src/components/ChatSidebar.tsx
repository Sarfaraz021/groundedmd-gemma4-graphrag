import { Plus, MessageSquare, Trash2, Sun, Moon, FlaskConical } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import IngestPanel from '@/components/IngestPanel';
import KnowledgeBaseSheet from '@/components/KnowledgeBaseSheet';
import type { ChatSession, IngestJob, Pipeline } from '@/lib/types';

interface Props {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  userName: string;
  ingestJobs: IngestJob[];
  pipelines?: Pipeline[];
  activePipelineId?: string | null;
  onSelectPipeline?: (id: string | null) => void;
}

const ChatSidebar = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  userName,
  ingestJobs,
  pipelines,
  activePipelineId,
  onSelectPipeline,
}: Props) => {
  const { theme, toggleTheme } = useTheme();
  return (
    <div className="w-[260px] h-screen flex flex-col bg-sidebar">
      {/* New chat */}
      <div className="p-4">
        <button
          onClick={onNewSession}
          className="w-full flex items-center gap-2 px-4 py-3 border border-sidebar-border text-[11px] font-semibold uppercase tracking-[0.12em] text-sidebar-accent-foreground hover:bg-sidebar-accent transition-colors"
        >
          <Plus className="w-4 h-4 text-primary" />
          New Chat
        </button>
      </div>

      <div className="px-4 pb-2 space-y-2">
        <KnowledgeBaseSheet activePipelineId={activePipelineId ?? null} />
      </div>
      <IngestPanel jobs={ingestJobs} />

      {/* Pipeline selector for ablation studies */}
      {onSelectPipeline && (
        <div className="px-4 pb-3 border-b border-sidebar-border">
          <div className="flex items-center gap-1.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-sidebar-foreground/50">
            <FlaskConical className="w-3 h-3" />
            Retrieval Pipeline
          </div>
          <div className="space-y-px">
            <button
              onClick={() => onSelectPipeline?.(null)}
              className={`w-full text-left px-2 py-1.5 text-[11px] transition-colors ${
                !activePipelineId
                  ? 'bg-sidebar-accent text-sidebar-accent-foreground font-semibold'
                  : 'text-sidebar-foreground hover:bg-sidebar-accent/50'
              }`}
            >
              All Pipelines
            </button>
            {(pipelines ?? []).map((p) => (
              <button
                key={p.id}
                onClick={() => onSelectPipeline?.(p.id)}
                className={`w-full text-left px-2 py-1.5 text-[11px] flex items-center justify-between gap-2 transition-colors ${
                  activePipelineId === p.id
                    ? 'bg-sidebar-accent text-sidebar-accent-foreground font-semibold'
                    : 'text-sidebar-foreground hover:bg-sidebar-accent/50'
                }`}
              >
                <span className="truncate">{p.id}</span>
                <span className="text-[10px] opacity-50 shrink-0">{p.doc_count}d</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Sessions */}
      <div className="flex-1 overflow-y-auto px-2 space-y-px">
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`group flex items-center gap-2.5 px-4 py-2.5 text-[13px] cursor-pointer transition-colors ${
              session.id === activeSessionId
                ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                : 'text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground'
            }`}
            onClick={() => onSelectSession(session.id)}
          >
            <MessageSquare className="w-3.5 h-3.5 shrink-0 opacity-40" />
            <span className="truncate flex-1">{session.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(session.id);
              }}
              className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-destructive transition-all shrink-0"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {/* User */}
      <div className="p-4 border-t border-sidebar-border space-y-3">
        <button
          onClick={toggleTheme}
          className="w-full flex items-center gap-2 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary flex items-center justify-center text-[11px] font-bold text-primary-foreground shrink-0">
            {userName[0]?.toUpperCase()}
          </div>
          <span className="text-[13px] text-sidebar-accent-foreground truncate flex-1">{userName}</span>
        </div>
      </div>
    </div>
  );
};

export default ChatSidebar;
