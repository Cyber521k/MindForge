import { useEffect, useRef, useState, useCallback } from "react";

export interface WSMessage {
  type: string;
  [key: string]: any;
}

/**
 * WebSocket client hook — connects to the MindForge sidecar on localhost:7878.
 * Automatically reconnects with backoff and buffers the last N messages.
 */
export function useWebSocket(url: string = "ws://localhost:7878/ws") {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<WSMessage[]>([]);
  const [latest, setLatest] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(3000);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectDelay.current = 3000;
        // Send a subscription message
        ws.send(JSON.stringify({ type: "subscribe", channels: ["*"] }));
      };

      ws.onclose = () => {
        setConnected(false);
        const delay = reconnectDelay.current;
        reconnectDelay.current = Math.min(delay * 1.5, 10000);
        setTimeout(connect, delay);
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
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: WSMessage) => {
    wsRef.current?.send(JSON.stringify(msg));
  }, []);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { connected, messages, latest, send, clearMessages };
}
