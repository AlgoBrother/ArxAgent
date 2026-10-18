import argparse
import json
import logging
from pathlib import Path
import mayatok

# Correctly formatted logging setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

INPUT_DIR = Path("data/extracted")
OUTPUT_DIR = Path("data/chunks")

def parse_args():
    parser = argparse.ArgumentParser(description="Chunk extracted paper JSONs using MayaTok BPE.")
    parser.add_argument("--vocab-path", type=str, default=None, help="Path to mayatok vocab JSON / model file")
    parser.add_argument("--model-name", type=str, default="vb100k", help="Preset tokenizer model name (e.g. vb100k)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=64, help="Token overlap between chunks")
    return parser.parse_args()

def chunk_with_mayatok(
    paper_data: dict,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
):
    paper_id = paper_data.get("paper_id", "unknown")
    pages = paper_data.get("pages", [])

    if not pages:
        return []

    full_text = "\n\n".join(
        [
            f"[PAGE {p['page']}]\n{p.get('full_text') or p.get('text', '')}"
            for p in pages
            if p.get("full_text") or p.get("text")
        ]
    ).strip()

    if not full_text:
        return []

    token_ids = tokenizer.encode(full_text)
    chunks = []
    start_idx = 0
    chunk_num = 0

    while start_idx < len(token_ids):
        end_idx = min(start_idx + max_tokens, len(token_ids))

        # Shrink until the independently encoded decoded text fits.
        while end_idx > start_idx:
            chunk_text = tokenizer.decode(token_ids[start_idx:end_idx]).strip()
            actual_count = len(tokenizer.encode(chunk_text))

            if chunk_text and actual_count <= max_tokens:
                break

            end_idx -= 1

        if end_idx <= start_idx:
            raise ValueError(
                f"Cannot produce a valid chunk for {paper_id} at token {start_idx}"
            )

        chunks.append({
            "paper_id": paper_id,
            "chunk_id": f"{paper_id}_C{chunk_num:04d}",
            "token_start": start_idx,
            "token_end": end_idx,
            "token_count": actual_count,
            "text": chunk_text,
        })
        chunk_num += 1

        if end_idx >= len(token_ids):
            break

        # Start the next chunk approximately overlap_tokens before the end.
        next_start = end_idx - overlap_tokens
        start_idx = max(start_idx + 1, next_start)

    return chunks

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize MayaTok instance from path or preset
    if args.vocab_path:
        tokenizer = mayatok.Tokenizer.load(args.vocab_path)
    elif hasattr(mayatok, "get_tokenizer"):
        tokenizer = mayatok.get_tokenizer(args.model_name)
    else:
        tokenizer = mayatok.Tokenizer.load_preset(args.model_name)

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        logging.warning(f"No JSON files found in {INPUT_DIR}")
        return

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                paper_data = json.load(f)

            chunks = chunk_with_mayatok(
                paper_data=paper_data,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap_tokens
            )

            out_file = OUTPUT_DIR / f"{json_path.stem}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

            logging.info(f"{json_path.stem}: Generated {len(chunks)} MayaTok chunks -> {out_file.name}")

        except Exception as e:
            logging.error(f"Failed to chunk {json_path.name}: {e}")

if __name__ == "__main__":
    main()