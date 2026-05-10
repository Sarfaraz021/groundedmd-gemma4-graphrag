import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Button } from '@/components/ui/button';

interface Props {
  children: ReactNode;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  message: string;
}

/**
 * Prevents a single component (e.g. parse preview) from taking down the whole chat shell.
 */
export default class PreviewErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: '' };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, message: error.message || 'Something went wrong.' };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ParsePreview]', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/95 p-6">
          <div className="max-w-md rounded-lg border border-border bg-card p-6 shadow-lg space-y-4">
            <p className="text-sm font-medium text-foreground">Preview crashed</p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {this.state.message} Try closing and uploading again, or use a smaller PDF. If this persists, open the
              browser console for details.
            </p>
            <Button
              type="button"
              onClick={() => {
                this.setState({ hasError: false, message: '' });
                this.props.onReset?.();
              }}
            >
              Dismiss
            </Button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
