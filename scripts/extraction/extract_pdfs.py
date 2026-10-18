
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import fitz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

INPUT_DIR = Path("data/papers")
OUTPUT_DIR = Path("data/extracted")


def order_blocks(blocks, page_width):
    """Order text blocks by columns for typical academic paper layouts."""
    text_blocks = []

    for b in blocks:
        if len(b) < 7 or b[6] != 0:
            continue

        text = b[4].strip()
        if text:
            text_blocks.append({
                "bbox": [round(v, 2) for v in b[:4]],
                "text": text,
            })

    if not text_blocks:
        return []

    # Detect whether blocks occupy both sides of the page.
    midpoint = page_width / 2
    left = []
    right = []

    for block in text_blocks:
        x0, y0, x1, y1 = block["bbox"]
        center_x = (x0 + x1) / 2

        if center_x < midpoint:
            left.append(block)
        else:
            right.append(block)

    # If both columns are populated, read down the left then down the right.
    if left and right:
        key = lambda b: (b["bbox"][1], b["bbox"][0])
        left.sort(key=key)
        right.sort(key=key)
        return left + right

    # Single-column page: ordinary top-to-bottom order.
    text_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return text_blocks


def extract_paper(pdf_path: Path) -> dict | None:
    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata or {}
            pages_data = []

            for i, page in enumerate(doc):
                blocks = order_blocks(
                    page.get_text("blocks"),
                    page.rect.width,
                )

                pages_data.append({
                    "page": i + 1,
                    "full_text": "\n\n".join(
                        block["text"] for block in blocks
                    ),
                    "blocks": blocks,
                })

            return {
                "paper_id": pdf_path.stem,
                "source_pdf": str(pdf_path),
                "metadata": {
                    "title": (meta.get("title") or "").strip(),
                    "author": (meta.get("author") or "").strip(),
                    "subject": (meta.get("subject") or "").strip(),
                    "keywords": (meta.get("keywords") or "").strip(),
                },
                "page_count": len(doc),
                "pages": pages_data,
            }

    except Exception:
        logging.exception("Failed to process %s", pdf_path.name)
        return None


def process_file(pdf_path: Path):
    data = extract_paper(pdf_path)
    if data is None:
        return

    out_path = OUTPUT_DIR / f"{data['paper_id']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info("Processed: %s -> %s", pdf_path.name, out_path.name)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        logging.warning("No PDF files found in %s", INPUT_DIR)
        return

    with ProcessPoolExecutor() as executor:
        list(executor.map(process_file, pdf_files))


if __name__ == "__main__":
    main()
