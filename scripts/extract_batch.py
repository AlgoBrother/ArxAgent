"""Resumable LLM extraction runner with schema/evidence checks and rate-limit stop."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts" / "extract_one.py"
SCHEMA_PATH = ROOT / "schemas" / "knowledge.schema.json"
CHUNKS_DIR = ROOT / "data" / "chunks"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge" / "by_chunk"
LOG_DIR = ROOT / "data" / "knowledge" / "batch_logs"
LOG_PATH = LOG_DIR / "latest.jsonl"

DEFAULT_MODEL = "qwen/qwen3.8-27b"
TIMEOUT_SECONDS = 300

KEEP_PATTERN = re.compile(
    r"(diverg|instabilit|unstable|spike|nan\b|warmup|warm-up|clip|ablat|"
    r"learning rate|batch size|we find|we observe|fail|limitation|"
    r"table\s+\d|figure\s+\d|result|experiment|outperform|"
    r"convergen|normalization|optimizer|optimization|scaling)",
    re.IGNORECASE,
)
RATE_LIMIT_MARKERS = (
    "rate_limit_exceeded", "ratelimiterror", "rate limit reached",
    "rate limit exceeded", "too many requests", "tokens per day",
    "tokens per minute", "http 429", "status code 429",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(file, **entry: Any) -> None:
    file.write(json.dumps({"timestamp": now_utc(), **entry}, ensure_ascii=False) + "\n")
    file.flush()


def load_papers() -> list[dict[str, str]]:
    path = ROOT / "bibliography.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "paper_id" not in rows[0]:
        raise RuntimeError("bibliography.csv must contain a paper_id column")
    return rows


def load_chunks(paper_id: str) -> list[dict[str, Any]]:
    path = CHUNKS_DIR / f"{paper_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                raise ValueError(f"Expected an object with a text field at {path}:{line_no}")

            expected_id = f"{paper_id}_C{len(chunks):04d}"
            actual_id = chunk.get("chunk_id", expected_id)
            if actual_id != expected_id:
                raise ValueError(
                    f"Chunk ID/order mismatch at {path}:{line_no}: "
                    f"expected {expected_id}, found {actual_id}"
                )
            chunk["chunk_id"] = expected_id
            chunks.append(chunk)
    return chunks


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_output(
    path: Path,
    paper_id: str,
    chunk_id: str,
    chunk_text: str,
    validator: Draft202012Validator,
) -> tuple[bool, str]:
    """Validate schema, paper/chunk identity, and evidence quotes against source text."""
    if not path.is_file():
        return False, "output file missing"
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON: {exc}"

    if not isinstance(data, dict):
        return False, "top-level value is not an object"
    if data.get("paper_id") != paper_id:
        return False, "paper_id mismatch"

    errors = list(validator.iter_errors(data))
    if errors:
        err = errors[0]
        where = "/".join(str(part) for part in err.absolute_path) or "<root>"
        return False, f"schema error at {where}: {err.message}"

    source = normalized(chunk_text)
    for ci, claim in enumerate(data.get("claims", []), start=1):
        for ei, evidence in enumerate(claim.get("evidence", []), start=1):
            if evidence.get("chunk_id") != chunk_id:
                return False, f"claim {ci} evidence {ei}: wrong chunk_id"
            quote = evidence.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                return False, f"claim {ci} evidence {ei}: empty/missing quote"
            if normalized(quote) not in source:
                return False, f"claim {ci} evidence {ei}: quote not found in chunk"
    return True, "schema and evidence passed"


def estimate_tokens(chunk_text: str, overhead: int) -> int:
    """Approximate input tokens; use only as a safety budget, not exact billing."""
    return math.ceil(len(chunk_text) / 4) + overhead


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable paper extraction with schema/evidence checks."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--paper", help="One paper ID, e.g. P20")
    selection.add_argument("--papers", help="Comma-separated paper IDs, e.g. P20,P22,P36")
    parser.add_argument("--max-chunks", type=int, help="Maximum new extraction attempts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID passed to extractor")
    parser.add_argument(
        "--keyword-filter", action="store_true",
        help="Skip chunks with fewer than two keyword matches (heuristic; off by default)",
    )
    parser.add_argument(
        "--token-budget", type=int,
        help="Approximate input-token budget for this run; disabled by default",
    )
    parser.add_argument(
        "--prompt-overhead-tokens", type=int, default=3500,
        help="Estimated fixed prompt/schema tokens per request (default: 3500)",
    )
    parser.add_argument(
        "--retry-invalid", action="store_true",
        help="Re-extract existing files that fail schema/evidence checks",
    )
    args = parser.parse_args()

    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be >= 1")
    if args.token_budget is not None and args.token_budget < 1:
        parser.error("--token-budget must be >= 1")
    if args.prompt_overhead_tokens < 0:
        parser.error("--prompt-overhead-tokens must be >= 0")
    if not EXTRACTOR.is_file():
        parser.error(f"Extractor not found: {EXTRACTOR}")
    if not SCHEMA_PATH.is_file():
        parser.error(f"Schema not found: {SCHEMA_PATH}")

    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
    except Exception as exc:
        parser.error(f"Could not load JSON Schema: {exc}")

    all_papers = load_papers()
    if args.paper:
        requested = [args.paper.strip()]
    elif args.papers:
        requested = [item.strip() for item in args.papers.split(",") if item.strip()]
        if not requested:
            parser.error("--papers must contain at least one paper ID")
    else:
        requested = [row["paper_id"] for row in all_papers]

    by_id = {row["paper_id"]: row for row in all_papers}
    unknown = [paper_id for paper_id in requested if paper_id not in by_id]
    if unknown:
        parser.error(f"Unknown paper ID(s): {', '.join(unknown)}")
    papers = [by_id[paper_id] for paper_id in requested]

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    attempted = succeeded = skipped = filtered = failed = estimated_used = 0
    stop_reason: str | None = None

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        for paper in papers:
            paper_id = paper["paper_id"]
            chunks = load_chunks(paper_id)

            for index, chunk in enumerate(chunks):
                chunk_id = chunk["chunk_id"]
                chunk_text = chunk["text"]
                output = KNOWLEDGE_DIR / paper_id / f"{chunk_id}.json"

                is_valid, reason = validate_output(
                    output, paper_id, chunk_id, chunk_text, validator
                )
                if is_valid:
                    skipped += 1
                    print(f"SKIP {chunk_id}: schema and evidence valid")
                    continue

                # Preserve an existing questionable file unless the user explicitly
                # authorizes replacing it; this prevents accidental data loss.
                if output.exists() and not args.retry_invalid:
                    skipped += 1
                    print(f"SKIP {chunk_id}: existing output rejected ({reason}); use --retry-invalid")
                    log_event(
                        log_file, paper_id=paper_id, chunk_id=chunk_id,
                        model=args.model, status="invalid_existing", reason=reason,
                    )
                    continue

                if args.keyword_filter and len(KEEP_PATTERN.findall(chunk_text)) < 2:
                    filtered += 1
                    print(f"FILTER {chunk_id}: keyword heuristic")
                    log_event(
                        log_file, paper_id=paper_id, chunk_id=chunk_id,
                        model=args.model, status="filtered",
                    )
                    continue

                request_estimate = estimate_tokens(chunk_text, args.prompt_overhead_tokens)
                if args.token_budget is not None and estimated_used + request_estimate > args.token_budget:
                    stop_reason = "estimated token budget reached"
                    print(
                        f"STOP: next request estimate ({request_estimate}) would exceed "
                        f"budget {args.token_budget}; estimated used={estimated_used}"
                    )
                    break
                if args.max_chunks is not None and attempted >= args.max_chunks:
                    stop_reason = "maximum new attempts reached"
                    break

                attempted += 1
                estimated_used += request_estimate
                print(
                    f"\n[{attempted}] Extracting {chunk_id} "
                    f"({index + 1}/{len(chunks)}; ~{request_estimate} input tokens)"
                )
                command = [
                    sys.executable, str(EXTRACTOR),
                    "--paper", paper_id,
                    "--chunk-index", str(index),
                    "--model", args.model,
                ]

                try:
                    result = subprocess.run(
                        command, cwd=ROOT, text=True, capture_output=True,
                        timeout=TIMEOUT_SECONDS, check=False,
                    )
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.stderr:
                        print(result.stderr, file=sys.stderr, end="")

                    output_text = ((result.stdout or "") + "\n" + (result.stderr or "")).casefold()
                    rate_limited = any(marker in output_text for marker in RATE_LIMIT_MARKERS)

                    if rate_limited:
                        failed += 1
                        status = "rate_limited"
                        stop_reason = "provider rate limit detected"
                        print(f"RATE LIMITED: {chunk_id}; stopping immediately.")
                    elif result.returncode == 0:
                        valid_after, why_after = validate_output(
                            output, paper_id, chunk_id, chunk_text, validator
                        )
                        if valid_after:
                            succeeded += 1
                            status = "success"
                        else:
                            failed += 1
                            status = "invalid_output"
                            print(f"INVALID OUTPUT: {chunk_id}: {why_after}")
                    else:
                        failed += 1
                        status = "failed"
                        print(f"FAILED: {chunk_id} (exit code {result.returncode})")

                    log_event(
                        log_file, paper_id=paper_id, chunk_id=chunk_id,
                        chunk_index=index, model=args.model, status=status,
                        return_code=result.returncode,
                        estimated_input_tokens=request_estimate,
                    )
                    if rate_limited:
                        break

                except subprocess.TimeoutExpired:
                    failed += 1
                    print(f"TIMEOUT: {chunk_id} after {TIMEOUT_SECONDS}s")
                    log_event(
                        log_file, paper_id=paper_id, chunk_id=chunk_id,
                        chunk_index=index, model=args.model, status="timeout",
                        estimated_input_tokens=request_estimate,
                    )
                except KeyboardInterrupt:
                    stop_reason = "interrupted by user"
                    print("\nInterrupted; saved outputs and log are preserved.")
                    log_event(
                        log_file, paper_id=paper_id, chunk_id=chunk_id,
                        chunk_index=index, model=args.model, status="interrupted",
                    )
                    break

            if stop_reason:
                break

    print("\n=== BATCH SUMMARY ===")
    print(f"Attempted:              {attempted}")
    print(f"Succeeded:              {succeeded}")
    print(f"Skipped:                {skipped}")
    print(f"Filtered:               {filtered}")
    print(f"Failed:                 {failed}")
    print(f"Estimated input tokens: {estimated_used}")
    print(f"Stop reason:            {stop_reason or 'completed selected papers'}")
    print(f"Log:                    {LOG_PATH.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
