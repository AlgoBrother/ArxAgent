import { NextRequest } from "next/server";

/**
 * The browser never sees ARXAGENT_BACKEND_URL or ARXAGENT_BACKEND_SECRET —
 * both are server-only env vars, so the Python backend's address and the
 * shared secret never reach client-side JS. This route is the single choke
 * point everything else (rate limiting, future auth) hangs off of.
 */

const BACKEND_URL = process.env.ARXAGENT_BACKEND_URL;
const BACKEND_SECRET = process.env.ARXAGENT_BACKEND_SECRET;

export async function POST(req: NextRequest) {
  if (!BACKEND_URL || !BACKEND_SECRET) {
    return new Response(
      JSON.stringify({ error: "Backend not configured on the server." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  let backendRes: Response;
  try {
    backendRes = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${BACKEND_SECRET}`,
      },
      body: JSON.stringify(body),
    });
  } catch {
    return new Response(JSON.stringify({ error: "Backend unreachable." }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!backendRes.ok || !backendRes.body) {
    const detail = await backendRes.text().catch(() => "");
    return new Response(
      JSON.stringify({ error: "Backend returned an error.", detail }),
      {
        status: backendRes.status || 502,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Pass the backend's streaming (SSE / chunked text) response straight
  // through to the browser without buffering it.
  return new Response(backendRes.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
