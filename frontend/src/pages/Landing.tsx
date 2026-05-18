import { useNavigate } from 'react-router-dom';

const features = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.955 11.955 0 003 12c0 2.757.9 5.376 2.496 7.506A11.955 11.955 0 0012 21.75c2.757 0 5.376-.9 7.504-2.494A11.955 11.955 0 0021 12c0-2.757-.9-5.376-2.496-7.506A11.959 11.959 0 0112 2.214z" />
      </svg>
    ),
    title: 'Every claim is cited',
    body: 'Answers trace back to exact graph nodes and source chunks. Press "show source" and read the passage yourself — no black box.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
      </svg>
    ),
    title: 'Local-first deployment',
    body: 'Gemma 4 26B MoE and embeddings run through Ollama, with Neo4j and FastAPI deployable on infrastructure you control.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />
      </svg>
    ),
    title: 'Graph-grounded retrieval',
    body: 'Documents are stored in a Neo4j knowledge graph. Retrieval traverses entities and relationships — not just keyword matches.',
  },
];

const stack = [
  { label: 'Gemma 4 26B MoE', sub: 'via Ollama · 25.8B params · local LLM' },
  { label: 'Neo4j GraphRAG', sub: 'knowledge graph' },
  { label: 'nomic-embed-text', sub: 'local embeddings' },
  { label: 'bge-reranker-v2-m3', sub: 'local cross-encoder' },
];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background text-foreground" style={{ fontFamily: "'Host Grotesk', sans-serif", letterSpacing: '-0.02em' }}>

      {/* Nav */}
      <nav className="border-b border-border">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="text-lg font-semibold tracking-tight">
            Grounded<span className="text-primary">MD</span>
          </span>
          <a
            href="https://github.com/Sarfaraz021/groundedmd-gemma4-graphrag"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            GitHub
          </a>
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-6xl mx-auto px-6 pt-20 pb-16 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/30 bg-primary/10 text-primary text-xs font-medium mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          Gemma 4 26B MoE · Ollama · Neo4j GraphRAG
        </div>

        <h1 className="text-5xl sm:text-6xl font-bold tracking-tight leading-tight mb-6 text-foreground">
          Clinical intelligence,<br />
          <span className="text-primary">local-first and grounded.</span>
        </h1>

        <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
          In a rural clinic 400 km from the nearest neurologist, a junior doctor faces a head-trauma case.
          The team needs evidence they can inspect, not a black-box answer.{' '}
          <span className="text-foreground">GroundedMD changes that.</span>
        </p>

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <button
            onClick={() => navigate('/app')}
            className="px-6 py-3 rounded-sm bg-primary text-primary-foreground font-semibold text-sm hover:opacity-90 transition-opacity"
          >
            Launch GroundedMD →
          </button>
          <button
            onClick={() => navigate('/ingest')}
            className="px-6 py-3 rounded-sm border border-primary/40 text-primary font-medium text-sm hover:bg-primary/10 transition-colors"
          >
            View Knowledge Base
          </button>
          <a
            href="#how-it-works"
            className="px-6 py-3 rounded-sm border border-border text-muted-foreground font-medium text-sm hover:text-foreground hover:border-primary/50 transition-colors"
          >
            How it works
          </a>
        </div>
      </section>

      {/* Demo preview */}
      <section className="max-w-6xl mx-auto px-6 pb-20">
        <div className="rounded-sm border border-border bg-card overflow-hidden">
          <div className="flex items-center gap-1.5 px-4 py-3 border-b border-border bg-sidebar-background">
            <span className="w-2.5 h-2.5 rounded-full bg-destructive/60" />
            <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
            <span className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
            <span className="ml-3 text-xs text-muted-foreground font-mono">GroundedMD · Gemma 4 26B MoE · local-first</span>
          </div>
          <div className="p-6 space-y-5" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
            <div className="flex gap-3 text-sm">
              <span className="text-primary shrink-0 font-medium">Doctor</span>
              <span className="text-foreground/80">Which blood-based biomarkers are discussed for acute TBI, and what roles do they play?</span>
            </div>
            <div className="flex gap-3 text-sm">
              <span className="text-green-400 shrink-0 font-medium">GroundedMD</span>
              <span className="text-muted-foreground">
                The corpus discusses biomarkers such as{' '}
                <span className="text-foreground font-medium">GFAP, UCH-L1, S100B, and NfL</span>
                {' '}as tools for diagnosis, injury characterization, prognosis, and monitoring.{' '}
                <span className="text-primary cursor-pointer">[1]</span>
                <br /><br />
                <span className="text-muted-foreground/60 text-xs">
                  [1] Anderson et al. · blood-biomarkers-ich-outcome-moderate-severe-tbi-anderson.pdf
                </span>
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="how-it-works" className="max-w-6xl mx-auto px-6 pb-20">
        <h2 className="text-2xl font-semibold text-center mb-12 text-foreground">
          Built for where it matters most
        </h2>
        <div className="grid sm:grid-cols-3 gap-5">
          {features.map((f) => (
            <div key={f.title} className="rounded-sm border border-border bg-card p-6">
              <div className="w-9 h-9 rounded-sm bg-primary/15 text-primary flex items-center justify-center mb-4">
                {f.icon}
              </div>
              <h3 className="font-semibold text-foreground mb-2">{f.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stack */}
      <section className="max-w-6xl mx-auto px-6 pb-20">
        <h2 className="text-2xl font-semibold text-center mb-10 text-foreground">Stack</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {stack.map((s) => (
            <div key={s.label} className="rounded-sm border border-border bg-card p-4 text-center">
              <p className="font-semibold text-foreground text-sm">{s.label}</p>
              <p className="text-xs text-muted-foreground mt-1">{s.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <div className="rounded-sm border border-primary/20 bg-primary/5 p-12 text-center">
          <h2 className="text-3xl font-bold mb-3 text-foreground">Try it now — no signup required</h2>
          <p className="text-muted-foreground mb-8 text-sm">Pre-loaded with evidence-based clinical TBI research papers. Ask anything.</p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              onClick={() => navigate('/app')}
              className="px-8 py-3 rounded-sm bg-primary text-primary-foreground font-semibold text-sm hover:opacity-90 transition-opacity"
            >
              Launch GroundedMD →
            </button>
            <button
              onClick={() => navigate('/ingest')}
              className="px-8 py-3 rounded-sm border border-primary/40 text-primary font-semibold text-sm hover:bg-primary/10 transition-colors"
            >
              View Knowledge Base
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-6 text-center text-xs text-muted-foreground">
        GroundedMD · Gemma 4 Good Hackathon 2026 · Apache 2.0
      </footer>
    </div>
  );
}
