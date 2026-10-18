"use client";

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";

type Paper = {
  title: string | null;
  pdf_url: string | null;
  authors: string[];
  published: string | null;
  summary: string | null;
  arxiv_id: string | null;
};

type Msg = {
  role: "user" | "assistant";
  content: string;
  steps?: string[];
  papers?: Paper[];
  done?: boolean;
};

const SESSION_KEY = "arxagent_session_id";

const EXAMPLES = [
  "Tell me about FlashAttention-2",
  "How do state space models differ from vanilla transformers?",
  "What problem did the original Transformer paper solve?",
];

// One id per tab so the backend keeps a separate ResearchAgent (history, memory) per conversation.
function getSessionId(): string {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

// Turn [2307.08691v1] / [id1, id2] / [github.com/x/y] into markdown links so they render as chips.
const ARXIV_ID = String.raw`\d{4}\.\d{4,5}(?:v\d+)?`;
const ARXIV_CITE = new RegExp(String.raw`\[(${ARXIV_ID}(?:\s*,\s*${ARXIV_ID})*)\]`, "g");
const REPO_CITE = /\[((?:github\.com|huggingface\.co)\/[^\]\s]+)\]/g;

function linkCitations(text: string): string {
  return text
    .replace(ARXIV_CITE, (_m, ids: string) =>
      ids
        .split(/\s*,\s*/)
        .map((id) => `[${id}](https://arxiv.org/abs/${id})`)
        .join(" ")
    )
    .replace(REPO_CITE, (_m, path: string) => `[${path}](https://${path})`);
}

const mdComponents = {
  a: ({ href, children }: { href?: string; children?: ReactNode }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={href?.startsWith("https://arxiv.org/abs/") ? "cite" : undefined}
    >
      {children}
    </a>
  ),
};

function Steps({ list }: { list: string[] }) {
  return (
    <ul className="steps">
      {list.map((s, i) => {
        const sub = /^\s{2,}/.test(s);
        return (
          <li key={i} className={`${sub ? "sub" : ""} ${i === list.length - 1 ? "now" : ""}`.trim()}>
            {s.trim()}
          </li>
        );
      })}
    </ul>
  );
}

function Trace({ steps, live }: { steps: string[]; live: boolean }) {
  if (live) {
    return (
      <div className="trace">
        <Steps list={steps.length ? steps : ["Starting"]} />
      </div>
    );
  }
  if (!steps.length) return null;
  return (
    <details className="trace">
      <summary>{steps.length} steps</summary>
      <Steps list={steps} />
    </details>
  );
}

function Sources({ papers }: { papers: Paper[] }) {
  return (
    <section className="sources" aria-label="Sources">
      <h2>Sources</h2>
      <ul>
        {papers.map((p, i) => {
          const href = p.arxiv_id ? `https://arxiv.org/abs/${p.arxiv_id}` : p.pdf_url ?? undefined;
          const authors = (p.authors ?? []).slice(0, 2).join(", ") + ((p.authors?.length ?? 0) > 2 ? " et al." : "");
          return (
            <li key={i} className="source">
              {href ? (
                <a href={href} target="_blank" rel="noreferrer">
                  {p.title ?? "Untitled"}
                </a>
              ) : (
                <span>{p.title ?? "Untitled"}</span>
              )}
              <div className="source-meta">
                <span>{authors}</span>
                <span>{p.published?.slice(0, 10)}</span>
              </div>
              {p.summary && <p>{p.summary}</p>}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Assistant({ m, streaming }: { m: Msg; streaming: boolean }) {
  const live = !m.content && !m.done;
  return (
    <article className="turn-ai">
      <Trace steps={m.steps ?? []} live={live} />
      {m.content && (
        <div className={`answer${streaming ? " streaming" : ""}`}>
          <ReactMarkdown components={mdComponents}>{linkCitations(m.content)}</ReactMarkdown>
        </div>
      )}
      {m.papers && m.papers.length > 0 && <Sources papers={m.papers} />}
    </article>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  // Keep the newest text in view, unless the reader has scrolled up.
  useEffect(() => {
    const onScroll = () => {
      stick.current = window.innerHeight + window.scrollY >= document.body.scrollHeight - 160;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (stick.current) endRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  }, [input]);

  function patchLast(fn: (m: Msg) => Msg) {
    setMessages((prev) => {
      if (!prev.length) return prev;
      const next = [...prev];
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });
  }

  function newChat() {
    controllerRef.current?.abort();
    sessionStorage.removeItem(SESSION_KEY);
    setMessages([]);
    setLoading(false);
    stick.current = true;
    taRef.current?.focus();
  }

  async function send(text?: string) {
    const query = (text ?? input).trim();
    if (!query || loading) return;

    stick.current = true;
    setMessages((m) => [...m, { role: "user", content: query }, { role: "assistant", content: "", steps: [] }]);
    setInput("");
    setLoading(true);

    const controller = new AbortController();
    controllerRef.current = controller;
    const current = () => controllerRef.current === controller;

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, session_id: getSessionId() }),
        signal: controller.signal,
      });

      if (res.status === 429) {
        patchLast((m) => ({ ...m, content: "You're asking too quickly. Wait a few seconds and try again." }));
        return;
      }
      if (!res.ok || !res.body) {
        patchLast((m) => ({ ...m, content: "Couldn't reach the research backend. Check that it's running, then try again." }));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames end with a blank line; keep any partial frame for the next chunk.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          let event: any;
          try {
            event = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }

          if (event.type === "step") {
            patchLast((m) => ({ ...m, steps: [...(m.steps ?? []), String(event.message)] }));
          } else if (event.type === "token") {
            patchLast((m) => ({ ...m, content: m.content + event.token }));
          } else if (event.type === "done" || event.type === "answer") {
            // Paths that never stream tokens (short replies, no evidence) carry the full text here.
            patchLast((m) => ({ ...m, content: m.content || event.answer || "", papers: event.papers ?? [] }));
          } else if (event.type === "error") {
            patchLast((m) => ({ ...m, content: `The agent hit an error: ${event.message}` }));
          }
        }
      }
    } catch (err) {
      if (!current()) return;
      const aborted = err instanceof DOMException && err.name === "AbortError";
      patchLast((m) => ({
        ...m,
        content: m.content || (aborted ? "Stopped." : "The connection dropped before an answer arrived. Try again."),
      }));
    } finally {
      if (current()) {
        setLoading(false);
        patchLast((m) => ({ ...m, done: true }));
      }
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <span className="wordmark">ArXAgent</span>
        {messages.length > 0 && (
          <button className="ghost" onClick={newChat}>
            New chat
          </button>
        )}
      </header>

      <main className="thread">
        {messages.length === 0 ? (
          <section className="empty">
            <h1>Ask about a paper, method, or model.</h1>
            <p>ArXAgent searches arXiv, Papers With Code and the web, then answers with citations you can open.</p>
            <ul className="prompts">
              {EXAMPLES.map((q) => (
                <li key={q}>
                  <button className="prompt" onClick={() => send(q)}>
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ) : (
          messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="turn-user">
                {m.content}
              </div>
            ) : (
              <Assistant key={i} m={m} streaming={loading && i === messages.length - 1} />
            )
          )
        )}
        <div ref={endRef} className="end" />
      </main>

      <footer className="composer">
        <div className="box">
          <textarea
            ref={taRef}
            rows={1}
            autoFocus
            value={input}
            aria-label="Your question"
            placeholder="Ask about a paper, method, or model"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
          />
          {loading ? (
            <button className="send stop" onClick={() => controllerRef.current?.abort()}>
              Stop
            </button>
          ) : (
            <button className="send" onClick={() => send()} disabled={!input.trim()}>
              Ask
            </button>
          )}
        </div>
        <p className="hint">Enter to send, Shift+Enter for a new line</p>
      </footer>
    </div>
  );
}