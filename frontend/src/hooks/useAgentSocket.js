import { useEffect, useRef, useState, useCallback } from "react";
import { api } from "../api";

/**
 * Shared WebSocket connection hook for both agents (docs/IMPLEMENTATION.md §8, §9).
 * Fans out incoming events by agent_id and maintains connection state.
 */
export function useAgentSocket({ sessionId = "default_session" } = {}) {
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const listenersRef = useRef(new Set());
  const reconnectTimeoutRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setConnecting(true);
    setError(null);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/chat`;

    try {
      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        setConnected(true);
        setConnecting(false);
        setError(null);
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          for (const listener of listenersRef.current) {
            listener(payload);
          }
        } catch (e) {
          console.error("Failed to parse websocket message:", e);
        }
      };

      socket.onerror = (e) => {
        console.warn("WebSocket error, will retry or fallback:", e);
        setError("WebSocket connection failed");
        setConnecting(false);
      };

      socket.onclose = () => {
        setConnected(false);
        setConnecting(false);
        wsRef.current = null;
        // Auto-reconnect after 3s
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };

      wsRef.current = socket;
    } catch (err) {
      console.warn("Could not create WebSocket:", err);
      setError(err.message);
      setConnecting(false);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback(
    async (message) => {
      const trimmed = (message || "").trim();
      if (!trimmed) return;

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            message: trimmed,
            session_id: sessionId,
          })
        );
      } else {
        // HTTP fallback if websocket is not open
        try {
          const res = await api.chatCompare(trimmed, sessionId);
          // Broadcast local results to listeners
          if (res) {
            if (res.agent1) {
              for (const listener of listenersRef.current) {
                listener({ agent_id: "agent1", event: "final", data: { answer: res.agent1.answer }, run_group_id: res.run_group_id });
                listener({ agent_id: "agent1", event: "metrics", data: res.agent1, run_group_id: res.run_group_id });
              }
            }
            if (res.agent2) {
              for (const listener of listenersRef.current) {
                listener({ agent_id: "agent2", event: "final", data: { answer: res.agent2.answer }, run_group_id: res.run_group_id });
                listener({ agent_id: "agent2", event: "metrics", data: res.agent2, run_group_id: res.run_group_id });
              }
            }
          }
        } catch (e) {
          console.error("HTTP chat fallback error:", e);
        }
      }
    },
    [sessionId]
  );

  const subscribe = useCallback((callback) => {
    listenersRef.current.add(callback);
    return () => {
      listenersRef.current.delete(callback);
    };
  }, []);

  return {
    connected,
    connecting,
    error,
    sendMessage,
    subscribe,
  };
}
