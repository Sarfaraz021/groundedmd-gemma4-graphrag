import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  message: string;
}

export default class AppErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: '' };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error?.message ?? 'Unknown error' };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[AppErrorBoundary] Unhandled UI error', error, info);
  }

  private reset = (): void => {
    this.setState({ hasError: false, message: '' });
  };

  render(): ReactNode {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6">
        <div className="max-w-xl w-full rounded-md border border-border bg-card p-5 space-y-3">
          <h1 className="text-base font-semibold">App crashed while rendering</h1>
          <p className="text-sm text-muted-foreground">
            {this.state.message || 'An unexpected UI error occurred.'}
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={this.reset}
              className="px-3 py-1.5 text-sm rounded border border-border hover:bg-muted"
            >
              Try again
            </button>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="px-3 py-1.5 text-sm rounded bg-primary text-primary-foreground hover:opacity-90"
            >
              Reload app
            </button>
          </div>
        </div>
      </div>
    );
  }
}

