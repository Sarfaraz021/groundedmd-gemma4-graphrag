import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import type { EvalScores } from '@/lib/types';

interface Props {
  scores: EvalScores;
}

const METRICS: { key: keyof EvalScores; label: string; description: string }[] = [
  { key: 'faithfulness',    label: 'Faithfulness',    description: 'Every claim grounded in retrieved chunks' },
  { key: 'completeness',    label: 'Completeness',    description: 'All relevant info from context covered' },
  { key: 'relevance',       label: 'Relevance',       description: 'Answer directly addresses the question' },
  { key: 'context_quality', label: 'Context Quality', description: 'Retrieved chunks were useful & sufficient' },
];

function scoreColor(score: number): string {
  if (score < 0) return 'text-muted-foreground';
  if (score >= 8) return 'text-emerald-500 dark:text-emerald-400';
  if (score >= 6) return 'text-amber-500 dark:text-amber-400';
  return 'text-red-500 dark:text-red-400';
}

function barColor(score: number): string {
  if (score < 0) return 'bg-muted';
  if (score >= 8) return 'bg-emerald-500 dark:bg-emerald-400';
  if (score >= 6) return 'bg-amber-500 dark:bg-amber-400';
  return 'bg-red-500 dark:bg-red-400';
}

function overallLabel(score: number): string {
  if (score < 0) return 'N/A';
  if (score >= 8) return 'Good';
  if (score >= 6) return 'Fair';
  return 'Poor';
}

export default function EvalScoresCard({ scores }: Props) {
  const [open, setOpen] = useState(false);
  const unavailable = scores.overall < 0;

  return (
    <div className="mt-3 rounded-lg border border-border bg-muted/30 text-xs overflow-hidden">
      {/* Clickable header row */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-muted/40 transition-colors"
      >
        <span className="font-semibold text-muted-foreground tracking-wide uppercase text-[10px]">
          Response Evaluation
        </span>
        <div className="flex items-center gap-2">
          {!unavailable && (
            <span className={`font-bold text-sm ${scoreColor(scores.overall)}`}>
              {scores.overall}/10 — {overallLabel(scores.overall)}
            </span>
          )}
          {unavailable && (
            <span className="text-muted-foreground">Unavailable</span>
          )}
          <ChevronDown
            className={`w-3.5 h-3.5 text-muted-foreground transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          />
        </div>
      </button>

      {/* Expandable detail section */}
      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-border">
          {!unavailable ? (
            <div className="space-y-2">
              {METRICS.map(({ key, label, description }) => {
                const score = scores[key] as number;
                return (
                  <div key={key}>
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="font-medium text-foreground/80" title={description}>
                        {label}
                      </span>
                      <span className={`font-bold ${scoreColor(score)}`}>{score}/10</span>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${barColor(score)}`}
                        style={{ width: `${score * 10}%` }}
                      />
                    </div>
                  </div>
                );
              })}
              {scores.reasoning && (
                <p className="mt-2 text-muted-foreground italic border-t border-border pt-2 leading-relaxed">
                  {scores.reasoning}
                </p>
              )}
            </div>
          ) : (
            <p className="text-muted-foreground italic">{scores.reasoning}</p>
          )}
        </div>
      )}
    </div>
  );
}
