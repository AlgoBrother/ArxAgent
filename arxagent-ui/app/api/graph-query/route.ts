import { NextRequest } from "next/server";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const url = process.env.ARXAGENT_BACKEND_URL;
  const secret = process.env.ARXAGENT_BACKEND_SECRET;
  if (!url || !secret) return Response.json({ error: "Graph chat backend is not configured." }, { status: 503 });
  let body: unknown;
  try { body = await request.json(); }
  catch { return Response.json({ error: "Invalid JSON body." }, { status: 400 }); }
  try {
    const upstream = await fetch(`${url.replace(/\/$/, "")}/graph-query`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${secret}`, "Accept": "text/event-stream" },
      body: JSON.stringify(body), cache: "no-store",
    });
    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text().catch(() => "");
      return Response.json({ error: text || `Backend returned ${upstream.status}` }, { status: upstream.status || 502 });
    }
    return new Response(upstream.body, { headers: { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no" } });
  } catch {
    return Response.json({ error: "Could not connect to the Python research API. Check that it is running." }, { status: 502 });
  }
}
