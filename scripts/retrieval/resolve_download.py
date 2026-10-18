import argparse, csv, os, re, time, requests
from difflib import SequenceMatcher
"""
Resolve and download papers from a bibliography.
""" 


API = "https://api.openalex.org/works"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bibliography.csv")
    ap.add_argument("--min-score", type=float, default=0.62)
    ap.add_argument("--delay", type=float, default=0.3)
    a = ap.parse_args()
    with open(a.csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("doi") and r.get("landing_url"):
            continue
        try:
            res = requests.get(
                API,
                params={
                    "search": r["title"],
                    "per-page": 5,
                    "select": "id,doi,title,publication_year,authorships,best_oa_location,open_access",
                },
                timeout=30,
                headers={"User-Agent": "ArxAgent/0.1"},
            )
            res.raise_for_status()
            works = res.json().get("results", [])
            if not works:
                r["status"] = "not_found"
                r["notes"] = "No OpenAlex candidate"
                continue
            works.sort(
                key=lambda w: SequenceMatcher(
                    None, norm(r["title"]), norm(w.get("title", ""))
                ).ratio(),
                reverse=True,
            )
            w = works[0]
            score = SequenceMatcher(
                None, norm(r["title"]), norm(w.get("title", ""))
            ).ratio()
            loc = w.get("best_oa_location") or {}
            pdf = loc.get("pdf_url") or ""
            landing = loc.get("landing_page_url") or w.get("doi") or w.get("id") or ""
            r["doi"] = (w.get("doi") or "").replace("https://doi.org/", "")
            r["landing_url"] = landing
            r["pdf_url"] = pdf
            r["authors_as_given"] = (
                "; ".join(
                    x.get("author", {}).get("display_name", "")
                    for x in w.get("authorships", [])[:6]
                )
                or r["authors_as_given"]
            )
            r["year"] = str(w.get("publication_year") or r["year"])
            if score < a.min_score:
                r["status"] = "needs_review"
                r["notes"] = f"Low title similarity {score:.3f}; verify manually"
            elif pdf:
                try:
                    os.makedirs("data/papers", exist_ok=True)
                    path = f'data/papers/{r["paper_id"]}.pdf'
                    p = requests.get(
                        pdf, timeout=90, headers={"User-Agent": "ArxAgent/0.1"}
                    )
                    p.raise_for_status()
                    if not p.content.startswith(b"%PDF"):
                        raise ValueError("URL did not return PDF bytes")
                    open(path, "wb").write(p.content)
                    r["local_pdf"] = path
                    r["status"] = "downloaded"
                    r["notes"] = f"Similarity {score:.3f}; verify canonical identity"
                except Exception as e:
                    r["status"] = "pdf_download_failed"
                    r["notes"] = str(e)
            else:
                r["status"] = "metadata_found_no_oa_pdf"
                r["notes"] = f"Similarity {score:.3f}; no OA PDF URL exposed"
            print(r["paper_id"], r["status"], r["title"])
            time.sleep(a.delay)
        except Exception as e:
            r["status"] = "resolver_error"
            r["notes"] = str(e)
            print(r["paper_id"], e)
    with open(a.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
