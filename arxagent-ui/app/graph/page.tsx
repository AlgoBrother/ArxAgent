"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FormEvent, useEffect, useRef, useState } from "react";
import "./graph-chat.css";

type Msg = {
  role: "user" | "assistant";
  content: string;
  steps?: string[];
  model?: string;
};

const STARTERS = [
  "How does DEEPNORM stabilize deep Transformers?",
  "What evidence links loss spikes to training instability?",
  "Compare interventions represented in the graph.",
];

function sessionId() {
  const key = "arxagent-graph-session";
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

export default function GraphChatPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function send(value?: string) {
    const query = (value ?? input).trim();
    if (!query || loading) return;

    const history = messages
      .slice(-8)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "", steps: [] },
    ]);
    setInput("");
    setError("");
    setLoading(true);

    const controller = new AbortController();
    abort.current = controller;

    const patch = (fn: (m: Msg) => Msg) =>
      setMessages((prev) => {
        const next = [...prev];
        if (next.length && next[next.length - 1].role === "assistant")
          next[next.length - 1] = fn(next[next.length - 1]);
        return next;
      });

    try {
      const res = await fetch("/api/graph-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, session_id: sessionId(), history }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        let message = `Request failed (${res.status})`;
        try {
          const j = await res.json();
          message = j.detail || j.error || message;
        } catch {}
        throw new Error(message);
      }

      const reader = res.body.getReader(),
        decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.split(/\r?\n/).find((x) => x.startsWith("data:"));
          if (!line) continue;
          let event: any;
          try {
            event = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }

          if (event.type === "step")
            patch((m) => ({
              ...m,
              steps: [...(m.steps ?? []), String(event.message)],
            }));
          else if (event.type === "answer")
            patch((m) => ({
              ...m,
              content: String(event.answer ?? ""),
              model: event.model,
            }));
          else if (event.type === "error")
            throw new Error(event.message || "Agent failed");
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError")
        patch((m) => ({ ...m, content: m.content || "Stopped." }));
      else {
        const message =
          e instanceof Error ? e.message : "Unexpected chat error";
        setError(message);
        patch((m) => ({
          ...m,
          content: m.content || `Investigation failed: ${message}`,
        }));
      }
    } finally {
      setLoading(false);
      abort.current = null;
      inputRef.current?.focus();
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    void send();
  }

  function reset() {
    abort.current?.abort();
    setMessages([]);
    setInput("");
    setError("");
    sessionStorage.removeItem("arxagent-graph-session");
  }

  return (
    <main className="graph-shell">
      <header className="graph-topbar">
        <a className="graph-brand" href="/">
          <span className="graph-mark">A</span>
          <span>
            ArxAgent <small>/ Graph Research</small>
          </span>
        </a>
        <div className="graph-top-actions">
          <span className="graph-status">
            <i /> LLMStability · FalkorDB
          </span>
          {messages.length > 0 && (
            <button className="graph-new" onClick={reset}>
              New chat
            </button>
          )}
        </div>
      </header>
      <section
        className={`graph-thread ${messages.length ? "has-messages" : ""}`}
      >
        {messages.length === 0 ? (
          <div className="graph-welcome">
            <div className="graph-eyebrow">
              <span /> RESEARCH AGENT · GRAPH-GROUNDED
            </div>
            <h1>
              Ask the graph.
              <br />
              <em>Follow the evidence.</em>
            </h1>
            <p>
              Investigate research claims across papers. ArxAgent follows
              connected claims and evidence in FalkorDB, then answers with
              source IDs, pages, and quotations.
            </p>
            <div className="graph-starters">
              {STARTERS.map((q) => (
                <button key={q} onClick={() => void send(q)}>
                  <span>{q}</span>
                  <b>↗</b>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="graph-messages">
            {messages.map((m, i) => (
              <article
                className={`graph-message ${m.role}`}
                key={`${i}-${m.role}`}
              >
                <div className="graph-avatar">
                  {m.role === "user" ? "YOU" : "AX"}
                </div>
                <div className="graph-message-body">
                  <div className="graph-message-label">
                    {m.role === "user" ? "You" : "ArxAgent"}
                  </div>
                  {m.role === "assistant" &&
                    loading &&
                    i === messages.length - 1 &&
                    !m.content && (
                      <div className="graph-thinking">
                        <span />
                        <span />
                        <span /> Investigating the graph
                      </div>
                    )}
                  {m.role === "assistant" && !!m.steps?.length && (
                    <details
                      className="graph-trace"
                      open={loading && i === messages.length - 1}
                    >
                      <summary>
                        Investigation trace · {m.steps.length} steps
                      </summary>
                      <ol>
                        {m.steps.map((s, j) => (
                          <li key={`${j}-${s}`}>{s}</li>
                        ))}
                      </ol>
                    </details>
                  )}
                  {m.content && (
                    <div className="graph-answer">
                      {m.role === "assistant" ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {m.content}
                        </ReactMarkdown>
                      ) : (
                        m.content
                      )}
                    </div>
                  )}
                  {m.model && (
                    <div className="graph-footnote">
                      Model: {m.model} · Limited to stored graph evidence
                    </div>
                  )}
                </div>
              </article>
            ))}
            <div ref={end} />
          </div>
        )}
      </section>
      {error && (
        <div className="graph-error" role="alert">
          {error}
        </div>
      )}
      <footer className="graph-composer-wrap">
        <form className="graph-composer" onSubmit={submit}>
          <textarea
            ref={inputRef}
            value={input}
            rows={1}
            disabled={loading}
            placeholder="Ask a research question…"
            aria-label="Research question"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                void send();
              }
            }}
          />
          {loading ? (
            <button
              type="button"
              className="graph-send stop"
              onClick={() => abort.current?.abort()}
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              className="graph-send"
              disabled={!input.trim()}
            >
              Investigate <span>↗</span>
            </button>
          )}
        </form>
        <div className="graph-disclaimer">
          <span>↳</span> Evidence-grounded answers. Missing evidence is
          disclosed, never invented.
        </div>
      </footer>
    </main>
  );
}