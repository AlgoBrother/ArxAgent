import argparse,json,os
ap=argparse.ArgumentParser(); ap.add_argument("--max-chars",type=int,default=9000); ap.add_argument("--overlap",type=int,default=800); a=ap.parse_args()
os.makedirs("data/chunks",exist_ok=True)
for fn in sorted(os.listdir("data/extracted")):
    if not fn.endswith(".json"): continue
    p=json.load(open("data/extracted/"+fn,encoding="utf-8"))
    text="\n\n".join(f'[PAGE {x["page"]}]\n{x["text"]}' for x in p["pages"] if x["text"])
    chunks=[]; start=0; n=0
    while start<len(text):
        end=min(start+a.max_chars,len(text))
        if end<len(text):
            cut=text.rfind("\n",start,end)
            if cut>start+a.max_chars//2: end=cut
        piece=text[start:end].strip()
        if piece: chunks.append({"paper_id":p["paper_id"],"chunk_id":f'{p["paper_id"]}_C{n:04d}',"char_start":start,"char_end":end,"text":piece}); n+=1
        if end>=len(text): break
        start=max(end-a.overlap,start+1)
    with open(f'data/chunks/{p["paper_id"]}.jsonl',"w",encoding="utf-8") as f:
        for c in chunks: f.write(json.dumps(c,ensure_ascii=False)+"\n")
    print(p["paper_id"],len(chunks),"chunks")
