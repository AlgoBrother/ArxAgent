import argparse
import json
import logging
from pathlib import Path

# Optional: pip install langchain-text-splitters / tiktoken for production token splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_DIR = Path("data/extracted")
OUTPUT_DIR = Path("data/chunks")

def parse_args():
    parser = argparse.ArgumentParser(description="Chunk extracted paper JSONs for RAG.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Target chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    return parser.parse_args()

def process_paper(json_path: Path, splitter: RecursiveCharacterTextSplitter):
    with open(json_path, encoding="utf-8") as f:
        paper = json.load(f)

    paper_id = paper.get("paper_id", json_path.stem)
    pages = paper.get("pages", [])

    if not pages:
        return

    # Build page-mapped text blocks to preserve origin tracking
    documents = []
    for page in pages:
        page_num = page.get("page")
        # Support both raw text and block structures
        page_text = page.get("full_text") or page.get("text") or ""
        if page_text.strip():
            documents.append({"text": page_text.strip(), "page": page_num})

    # Consolidate text while keeping track of page ranges
    full_text = "\n\n".join([f"[PAGE {d['page']}]\n{d['text']}" for d in documents])

    # Recursive split prevents breaking sentences mid-way
    raw_chunks = splitter.split_text(full_text)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "paper_id": paper_id,
            "chunk_id": f"{paper_id}_C{idx:04d}",
            "text": chunk_text.strip(),
            "char_len": len(chunk_text)
        })

    # Save as JSONL
    out_file = OUTPUT_DIR / f"{paper_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logging.info(f"{paper_id}: Generated {len(chunks)} chunks -> {out_file.name}")

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Standard recursive splitter respecting paragraph and sentence boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        separators=["\n[PAGE ", "\n\n", "\n", ". ", " ", ""]
    )

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        logging.warning(f"No extraction files found in {INPUT_DIR}")
        return

    for json_path in json_files:
        try:
            process_paper(json_path, splitter)
        except Exception as e:
            logging.error(f"Failed to process {json_path.name}: {e}")

if __name__ == "__main__":
    main()