import { NextRequest, NextResponse } from "next/server";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

/**
 * Per-IP request limiter guarding every /api/* route, sitting in front of the
 * model rate limiter that lives on the Python backend. This one is cheap and
 * stops raw request-flooding before it ever reaches Groq.
 *
 * Production (Vercel): set UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN
 * (free tier at upstash.com) so limits are shared across every serverless
 * instance handling requests.
 *
 * Local dev without those env vars: falls back to an in-memory counter.
 * This fallback is NOT safe in production — a deployed app can run multiple
 * instances with separate memory, so it would not actually enforce a global
 * limit. It exists purely so `next dev` works out of the box.
 */

const WINDOW = "1 m";
const LIMIT = 20; // requests per IP per window

const hasUpstash =
  !!process.env.UPSTASH_REDIS_REST_URL && !!process.env.UPSTASH_REDIS_REST_TOKEN;

const ratelimit = hasUpstash
  ? new Ratelimit({
      redis: new Redis({
        url: process.env.UPSTASH_REDIS_REST_URL!,
        token: process.env.UPSTASH_REDIS_REST_TOKEN!,
      }),
      limiter: Ratelimit.slidingWindow(LIMIT, WINDOW),
      analytics: true,
      prefix: "arxagent",
    })
  : null;

const memoryHits = new Map<string, { count: number; reset: number }>();

function memoryLimit(ip: string) {
  const now = Date.now();
  const windowMs = 60_000;
  const entry = memoryHits.get(ip);
  if (!entry || now > entry.reset) {
    memoryHits.set(ip, { count: 1, reset: now + windowMs });
    return { success: true };
  }
  entry.count += 1;
  return { success: entry.count <= LIMIT };
}

export async function proxy(req: NextRequest) {
  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ??
    req.headers.get("x-real-ip") ??
    "127.0.0.1";

  const result = ratelimit ? await ratelimit.limit(ip) : memoryLimit(ip);

  if (!result.success) {
    return NextResponse.json(
      { error: "Too many requests. Please slow down and try again shortly." },
      { status: 429, headers: { "Retry-After": "60" } }
    );
  }

  return NextResponse.next();
}

export const config = {
  matcher: "/api/:path*",
};