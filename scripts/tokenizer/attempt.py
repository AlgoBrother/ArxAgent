
import json
import re
from pathlib import Path
from jsonschema import Draft202012Validator

ROOT = Path(".")
SCHEMA_PATH = ROOT / "schemas/knowledge.schema.json"
OUT_ROOT = ROOT / "data/knowledge/by_chunk"

schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
validator = Draft202012Validator(schema)


def norm(s):
    return " ".join(s.split()).casefold()


def load_chunks(paper_id):
    path = ROOT / "data/chunks" / f"{paper_id}.jsonl"
    return {
        item["chunk_id"]: item["text"]
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def find_page(text, quote):
    # Split the chunk at page markers, preserving page numbers.
    matches = list(re.finditer(r"\[PAGE\s+(\d+)\]", text))
    target = norm(quote)

    for i, marker in enumerate(matches):
        start = marker.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_text = text[start:end]

        if target in norm(page_text):
            return int(marker.group(1))

    return None


candidates = [
    {
        "paper_id": "P05",
        "paper_title": "Small-scale proxies for large-scale Transformer training instabilities",
        "chunk_id": "P05_C0000",
        "claim_id": "P05_MANUAL_001",
        "claim_type": "mechanism",
        "statement": "The authors investigate reproducing and studying large-scale Transformer training instabilities at smaller scales.",
        "epistemic_status": "proposed",
        "confidence": 0.95,
        "quote": "we seek ways to reproduce and study training stability and instability at smaller scales.",
    },
    {
        "paper_id": "P07",
        "paper_title": "Spike No More: Stabilizing the Pre-training of Large Language Models",
        "chunk_id": "P07_C0000",
        "claim_id": "P07_MANUAL_001",
        "claim_type": "failure_mode",
        "statement": "Loss spikes can degrade large language model performance and sometimes ruin pretraining.",
        "epistemic_status": "observed",
        "confidence": 0.95,
        "quote": "The spikes degrade the performance of large language models and sometimes ruin the pre-training.",
    },
    {
        "paper_id": "P12",
        "paper_title": "Stabilizing Transformer Training by Preventing Attention Entropy Collapse",
        "chunk_id": "P12_C0000",
        "claim_id": "P12_MANUAL_001",
        "claim_type": "experimental_result",
        "statement": "The authors report an association between low attention entropy and high training instability.",
        "epistemic_status": "observed",
        "confidence": 0.95,
        "quote": "low attention entropy is accompanied by high training instability",
    },
    {
        "paper_id": "P27",
        "paper_title": "Why Warmup the Learning Rate? Underlying Mechanisms and Improvements",
        "chunk_id": "P27_C0000",
        "claim_id": "P27_MANUAL_001",
        "claim_type": "mechanism",
        "statement": "The authors report that tolerating a larger target learning rate improves hyperparameter robustness and final performance.",
        "epistemic_status": "observed",
        "confidence": 0.90,
        "quote": "The ability to handle larger ηtrgt makes hyperparameter tuning more robust while improving the final performance.",
    },
    {
        "paper_id": "P39",
        "paper_title": "Mixed Precision Training",
        "chunk_id": "P39_C0000",
        "claim_id": "P39_MANUAL_001",
        "claim_type": "experimental_result",
        "statement": "The authors introduce half-precision training intended to reduce memory requirements without losing model accuracy.",
        "epistemic_status": "proposed",
        "confidence": 0.90,
        "quote": "We introduce methodology for training deep neural networks using half-precision floating point numbers, without losing model accuracy",
    },
]

for item in candidates:
    paper_id = item["paper_id"]
    chunk_id = item["chunk_id"]
    quote = item["quote"]

    try:
        chunks = load_chunks(paper_id)
    except FileNotFoundError:
        print(f"SKIP {chunk_id}: chunk file missing")
        continue

    text = chunks.get(chunk_id)
    if text is None:
        print(f"SKIP {chunk_id}: chunk missing")
        continue

    if norm(quote) not in norm(text):
        print(f"SKIP {chunk_id}: quote not found; inspect source text")
        continue

    page = find_page(text, quote)
    if page is None:
        print(f"SKIP {chunk_id}: quote not found within a page section")
        continue

    record = {
        "paper_id": paper_id,
        "paper_title": item["paper_title"],
        "claims": [{
            "claim_id": item["claim_id"],
            "claim_type": item["claim_type"],
            "statement": item["statement"],
            "epistemic_status": item["epistemic_status"],
            "confidence": item["confidence"],
            "failure_modes": (
                ["loss spikes"] if paper_id == "P07" else
                ["training instability"] if paper_id in ("P05", "P12") else []
            ),
            "interventions": [],
            "metrics": [],
            "evidence": [{
                "quote": quote,
                "page": page,
                "section": None,
                "chunk_id": chunk_id,
            }],
        }],
    }

    errors = list(validator.iter_errors(record))
    if errors:
        print(f"SKIP {chunk_id}: schema error: {errors[0].message}")
        continue

    # Prevent accidental overwrite of an existing record.
    out_path = OUT_ROOT / paper_id / f"{chunk_id}.json"
    if out_path.exists():
        print(f"SKIP {chunk_id}: output already exists: {out_path}")
        continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"CREATED {out_path} (page {page})")
