# ArxAgent Paper Extraction Pipeline

Starter scaffold for the 40-paper Transformer/LLM training-stability corpus.

## Pipeline
1. Resolve candidate metadata and download PDFs when an open-access PDF URL is available.
2. Extract page-indexed text.
3. Make overlapping chunks that preserve page markers.
4. Run ArxAgent/your configured LLM with `prompts/extract_claims.md`.
5. Validate JSON and manually review evidence.
6. Import reviewed records into FalkorDB.

## Setup
Python 3.11+ recommended.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/resolve_download.py --csv bibliography.csv
python scripts/extract_pdfs.py
python scripts/make_chunks.py --max-chars 9000 --overlap 800
```
The resolver uses OpenAlex and only downloads PDFs it identifies as open access. Candidate title similarity is not proof of identity; verify the CSV before relying on a match. It does not bypass paywalls and won't necessarily find all 40 PDFs.

Run extraction using your existing ArxAgent LLM layer: send each chunk to the prompt and save one validated JSON file per paper in `data/knowledge/P01.json`, etc. See `schemas/knowledge.schema.json`.

```bash
python scripts/validate_knowledge.py
export FALKORDB_HOST=localhost
export FALKORDB_PORT=6379
python scripts/ingest_falkordb.py --graph LLMStability
```

FalkorDB uses Redis/RESP on port **6379**; port **3000** is the browser UI.

## Graph model
`Paper -> Claim -> FailureMode / Intervention / Metric`
Claims carry `epistemic_status` (`observed`, `hypothesized`, `demonstrated`, `author_recommendation`, `unclear`) and page-level evidence quotes. Don't auto-infer contradictions across papers; require review.

## Quality gates
- Verify canonical title, authors, year and DOI/arXiv ID.
- Preserve exact page evidence for each claim.
- Do not upgrade hypotheses to causal demonstrations.
- Record negative results and trade-offs.
- Treat PDF tables/equations carefully; inspect important evidence manually.
