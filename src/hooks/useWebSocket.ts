import { useEffect, useRef, useState, useCallback } from "react";

export interface WSMessage {
  type: string;
  [key: string]: any;
}

/**
 * WebSocket client hook — connects to the MindForge sidecar on localhost:7878.
 * Automatically reconnects with exponential backoff + jitter and buffers the last N messages.
 */
export function useWebSocket(url: string = "ws://localhost:7878/ws") {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<WSMessage[]>([]);
  const [latest, setLatest] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(3000);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 20;
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);

  const connect = useCallback(() => {
    if (unmountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (reconnectAttempts.current >= maxReconnectAttempts) return;
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectDelay.current = 3000;
        reconnectAttempts.current = 0;
        ws.send(JSON.stringify({ type: "subscribe", channels: ["*"] }));
      };

      ws.onclose = () => {
        setConnected(false);
        if (unmountedRef.current) return;
        reconnectAttempts.current += 1;
        if (reconnectAttempts.current >= maxReconnectAttempts) {
          console.warn("[useWebSocket] Max reconnection attempts reached, giving up");
          return;
        }
        // Exponential backoff with jitter
        const base = reconnectDelay.current;
        const jitter = Math.random() * 1000;
        const delay = Math.min(base + jitter, 15000);
        reconnectDelay.current = Math.min(base * 1.5, 15000);
        reconnectTimer.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onmessage = (e) => {
        try {
          const msg: WSMessage = JSON.parse(e.data);
          setLatest(msg);
          setMessages((prev) => [...prev.slice(-199), msg]);
        } catch {
          // Ignore non-JSON messages
        }
      };
    } catch {
      // WebSocket constructor can throw in some environments
    }
  }, [url]);

  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: WSMessage) => {
    wsRef.current?.send(JSON.stringify(msg));
  }, []);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { connected, messages, latest, send, clearMessages };
}
