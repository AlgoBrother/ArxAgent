import os,json,fitz
os.makedirs("data/extracted",exist_ok=True)
for fn in sorted(os.listdir("data/papers")):
    if not fn.lower().endswith(".pdf"): continue
    doc=fitz.open(os.path.join("data/papers",fn))
    obj={"paper_id":os.path.splitext(fn)[0],"source_pdf":f"data/papers/{fn}","page_count":len(doc),
         "pages":[{"page":i+1,"text":p.get_text("text",sort=True).strip()} for i,p in enumerate(doc)]}
    out=f'data/extracted/{obj["paper_id"]}.json'
    json.dump(obj,open(out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(fn,"->",out)
