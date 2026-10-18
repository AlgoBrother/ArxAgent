"""api.py: HTTP entry point for the graph agent. Same SSE contract the
Next.js UI already uses (/graph-query). Run from the repo root:

    export ARXAGENT_BACKEND_SECRET=$(openssl rand -hex 32)
    uvicorn api:app --port 8000
"""
import hmac, json, logging, os, queue, re, threading, time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from scripts import research_agent as agent
from scripts.graph.graph_tools import DEFAULT_GRAPH

log = logging.getLogger("arxagent.api")
app = FastAPI()

SECRET = os.getenv("ARXAGENT_BACKEND_SECRET", "")
CACHE_PATH = ROOT / "data" / "demo_cache.json"
PER_IP_PER_HOUR = int(os.getenv("PER_IP_PER_HOUR", "4"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "12"))
MAX_ROUNDS = int(os.getenv("API_MAX_ROUNDS", "4"))

_hits: dict[str, deque] = defaultdict(deque)
_daily = {"day": "", "n": 0}
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(2)  # at most 2 agent runs at once

FRIENDLY = {
    "diagnose": "Looking up reported failures and fixes",
    "search_claims": "Searching claims and linked evidence",
    "get_claim": "Inspecting a claim and its evidence",
    "explore_concept": "Exploring connected concepts",
    "related_claims": "Following graph links to related claims",
    "find_papers": "Searching arXiv for papers the graph lacks",
}


class ChatBody(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    session_id: str | None = None
    history: list[dict] = Field(default_factory=list, max_length=12)  # accepted, not used yet


def check_auth(authorization: str | None) -> None:
    if not SECRET:
        raise HTTPException(503, "ARXAGENT_BACKEND_SECRET is not configured")
    if not hmac.compare_digest(authorization or "", f"Bearer {SECRET}"):
        raise HTTPException(401, "Unauthorized")


def sse(kind: str, **data) -> str:
    return "data: " + json.dumps({"type": kind, **data}, ensure_ascii=False, default=str) + "\n\n"


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check_limits(ip: str) -> None:
    now, today = time.time(), date.today().isoformat()
    with _lock:
        if _daily["day"] != today:
            _daily.update(day=today, n=0)
        if _daily["n"] >= DAILY_CAP:
            raise HTTPException(429, "Today's demo budget is used up. Try the example questions.")
        q = _hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= PER_IP_PER_HOUR:
            raise HTTPException(429, "Too many requests. Please wait a while.",
                                headers={"Retry-After": "600"})
        q.append(now)
        _daily["n"] += 1


def friendly(msg: str) -> str:
    m = re.match(r"Round \d+: (\w+) ->", msg)
    return FRIENDLY.get(m.group(1), "Running a graph tool") if m else msg


def stream(events):
    headers = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
    return StreamingResponse(events, media_type="text/event-stream", headers=headers)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph-examples")
def graph_examples():
    return {"questions": [v["question"] for v in load_cache().values()]}


@app.post("/graph-query")
def graph_query(body: ChatBody, request: Request,
                authorization: str | None = Header(default=None)):
    check_auth(authorization)
    question = body.query.strip()

    cached = load_cache().get(norm(question))
    if cached:  # real saved run: free, no limits
        r = cached["response"]
        def replay():
            yield sse("step", message="Loaded a saved example run")
            yield sse("answer", answer=r["answer_markdown"], sources=r.get("sources", []),
                      trace_summary=r.get("trace_summary", []), model=r.get("model"),
                      graph=DEFAULT_GRAPH, cached=True)
        return stream(replay())

    check_limits(request.client.host if request.client else "unknown")
    if not _slots.acquire(timeout=1):
        raise HTTPException(429, "The agent is busy. Please retry in a few seconds.")

    q: "queue.Queue" = queue.Queue()
    DONE = object()

    def worker():
        try:
            out = agent.answer_question(
                question, max_rounds=MAX_ROUNDS, max_completion_tokens=800,
                session_id=None, allow_ingest=False,
                progress=lambda m: q.put(("step", {"message": friendly(m)})))
            q.put(("answer", {"answer": out["answer_markdown"], "sources": out["sources"],
                              "trace_summary": out["trace_summary"], "model": out["model"],
                              "graph": DEFAULT_GRAPH, "cached": False}))
        except agent.AgentError as exc:
            q.put(("error", {"message": str(exc)}))
        except Exception as exc:
            log.exception("agent run failed")
            msg = ("The demo's model budget is used up for now. Try the example questions."
                   if exc.__class__.__name__ == "RateLimitError"
                   else "The agent failed. Please try again.")
            q.put(("error", {"message": msg}))
        finally:
            _slots.release()
            q.put((DONE, None))

    threading.Thread(target=worker, daemon=True).start()

    def events():
        yield sse("step", message="Connected to the graph. Planning the investigation…")
        while True:
            kind, data = q.get()
            if kind is DONE:
                return
            yield sse(kind, **data)

    return stream(events())