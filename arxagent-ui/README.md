# ArXAgent UI

Next.js frontend for ArXAgent. Talks to your existing Python backend only
through the server-side proxy at `app/api/query/route.ts` — the backend URL
and shared secret are server-only env vars and never reach the browser.

## Setup

```bash
npm install
cp .env.example .env.local   # fill in ARXAGENT_BACKEND_URL and ARXAGENT_BACKEND_SECRET
npm run dev
```

## Architecture

```
Browser → Next.js API route (/api/query) → Python backend (/query)
              ↑ per-IP rate limit              ↑ Bearer secret required
           (middleware.ts)                  (your model rate limiter goes here)
```

- **middleware.ts** — per-IP request limiter on every `/api/*` route, using
  Upstash Redis if `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` are
  set, else an in-memory fallback for local dev only.
- **app/api/query/route.ts** — the only thing that ever calls your Python
  backend. Adds the `Authorization: Bearer <secret>` header and streams the
  backend's response straight through to the browser.
- **app/page.tsx** — minimal chat UI that reads the streamed response
  incrementally.

## What the Python backend needs to expose

A `POST /query` endpoint that:
1. Rejects requests without a valid `Authorization: Bearer <ARXAGENT_BACKEND_SECRET>`
   header (401).
2. Accepts `{ "query": string }` as JSON body (extend as needed — e.g. a
   session id for ChromaDB memory).
3. Streams its response back as plain text or SSE chunks (`text/event-stream`
   or chunked `text/plain` both work — the proxy route passes the body
   through unmodified).
4. Enforces your own model-level rate limiter (token bucket, per-session +
   global ceiling) before calling Groq, and returns 429 on limit-hit so the
   UI can surface it.

CORS is not needed on the Python side since the browser only ever calls
your own Next.js origin — the Python backend only needs to trust requests
carrying the shared secret.

## Deploying

- **Frontend**: Vercel free tier. Set `ARXAGENT_BACKEND_URL`,
  `ARXAGENT_BACKEND_SECRET`, and the two `UPSTASH_*` vars in the project's
  Environment Variables.
- **Backend**: keep it on whatever you choose (Render/Fly/HF Spaces) — just
  make sure `ARXAGENT_BACKEND_SECRET` matches on both sides.
