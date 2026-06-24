import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * React Error Boundary — catches render errors and shows a fallback UI
 * with a retry button instead of a blank white screen.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          background: "var(--bg)",
          color: "var(--text)",
          padding: 40,
          textAlign: "center",
        }}>
          <div style={{ fontSize: 48, color: "var(--error)", marginBottom: 16 }}>⚠</div>
          <h1 style={{ fontSize: 22, color: "var(--error)", marginBottom: 8 }}>
            Something went wrong
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: 14, marginBottom: 20, maxWidth: 500 }}>
            {this.state.error?.message || "An unexpected error occurred while rendering this screen."}
          </p>
          <button
            onClick={this.handleReset}
            className="btn-gold gold-glow"
            style={{ padding: "12px 24px", fontSize: 16 }}
          >
            ↻ Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
