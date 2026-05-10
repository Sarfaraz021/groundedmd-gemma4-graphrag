import { useState, useCallback } from 'react';
import type { ChatSession, ChatMessage, IngestJob, IngestStreamEvent, IngestMeta, GraphPreview } from '@/lib/types';

const SESSIONS_KEY = 'chat_sessions';

function loadSessions(): ChatSession[] {
  try {
    const stored = localStorage.getItem(SESSIONS_KEY);
    if (!stored) return [];
    const parsed: unknown = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((item: unknown) => {
      const s = item as Record<string, unknown>;
      return {
        ...s,
        createdAt: new Date(s.createdAt as string | number | Date),
        messages: Array.isArray(s.messages)
          ? (s.messages as Record<string, unknown>[]).map((m) => ({
              ...m,
              timestamp: new Date(m.timestamp as string | number | Date),
              sourceChunks: m.sourceChunks as ChatMessage["sourceChunks"],
            }))
          : [],
      } as ChatSession;
    });
  } catch {
    console.warn('[useChatSessions] localStorage read failed — resetting chat sessions.');
    return [];
  }
}

function saveSessions(sessions: ChatSession[]) {
  try {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  } catch {
    // QuotaExceededError or SecurityError — persist what we can, never crash the app
    console.warn('[useChatSessions] localStorage write failed — session not persisted.');
  }
}

export function useChatSessions() {
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(
    () => sessions[0]?.id ?? null
  );

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;

  const createSession = useCallback(() => {
    const session: ChatSession = {
      id: crypto.randomUUID(),
      title: 'New Chat',
      createdAt: new Date(),
      messages: [],
      ingestJobs: [],
    };
    setSessions((prev) => {
      const next = [session, ...prev];
      saveSessions(next);
      return next;
    });
    setActiveSessionId(session.id);
    return session.id;
  }, []);

  const deleteSession = useCallback((id: string) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      saveSessions(next);
      return next;
    });
    setActiveSessionId((current) => (current === id ? null : current));
  }, []);

  const addMessage = useCallback((sessionId: string, message: ChatMessage) => {
    setSessions((prev) => {
      const next = prev.map((s) => {
        if (s.id !== sessionId) return s;
        const messages = [...s.messages, message];
        const title = s.messages.length === 0 ? message.content.slice(0, 40) : s.title;
        return { ...s, messages, title };
      });
      saveSessions(next);
      return next;
    });
  }, []);

  const addIngestJob = useCallback((sessionId: string, job: IngestJob) => {
    setSessions((prev) => {
      const next = prev.map((s) => {
        if (s.id !== sessionId) return s;
        const jobs = [...(s.ingestJobs ?? []), job];
        return { ...s, ingestJobs: jobs };
      });
      saveSessions(next);
      return next;
    });
  }, []);

  const appendIngestEvent = useCallback((sessionId: string, jobId: string, ev: IngestStreamEvent) => {
    // Events are live/ephemeral — do NOT persist to localStorage (causes
    // QuotaExceededError with bulk ingests that emit hundreds of events per file).
    // Only the final job status (set by finishIngestJob) is persisted.
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s;
        const jobs = (s.ingestJobs ?? []).map((j) => {
          if (j.id !== jobId) return j;
          const events = [...j.events, ev].slice(-400);
          return { ...j, events };
        });
        return { ...s, ingestJobs: jobs };
      })
    );
  }, []);

  const setIngestJobMeta = useCallback((sessionId: string, jobId: string, meta: IngestMeta) => {
    // Meta is live state — not persisted (same reason as appendIngestEvent).
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s;
        const jobs = (s.ingestJobs ?? []).map((j) => (j.id === jobId ? { ...j, meta } : j));
        return { ...s, ingestJobs: jobs };
      })
    );
  }, []);

  const finishIngestJob = useCallback(
    (sessionId: string, jobId: string, payload: { status: 'complete' | 'error'; graphPreview?: GraphPreview; error?: string }) => {
      setSessions((prev) => {
        const next = prev.map((s) => {
          if (s.id !== sessionId) return s;
          const jobs = (s.ingestJobs ?? []).map((j) => {
            if (j.id !== jobId) return j;
            const { graphPreview, ...rest } = payload;
            return {
              ...j,
              ...rest,
              ...(graphPreview !== undefined ? { graphPreview } : {}),
            };
          });
          return { ...s, ingestJobs: jobs };
        });
        saveSessions(next);
        return next;
      });
    },
    []
  );

  return {
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
  };
}
