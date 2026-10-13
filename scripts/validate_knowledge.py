import glob,json,sys
from jsonschema import Draft202012Validator
schema=json.load(open("schemas/knowledge.schema.json",encoding="utf-8")); v=Draft202012Validator(schema)
files=glob.glob("data/knowledge/*.json")
if not files: print("No JSON files in data/knowledge"); sys.exit(2)
bad=0
for f in files:
    errors=list(v.iter_errors(json.load(open(f,encoding="utf-8"))))
    if errors:
        bad+=1; print("FAIL",f)
        for e in errors: print(" -",list(e.absolute_path),e.message)
    else: print("OK",f)
sys.exit(1 if bad else 0)
