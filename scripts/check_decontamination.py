import json
tulupath = "/data/niklas/gritlm/tuluv2.jsonl"
gsmpath = "/data/niklas/gritlm/evaldata/eval/gsm/test.jsonl"

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f.readlines()]

tulu = load_jsonl(tulupath)
gsm = load_jsonl(gsmpath)

cont = 0

for t in gsm:
    for t2 in tulu:
        if any(t['answer'].strip() in _ for _ in t2['text']):
            cont += 1
            break
    
cont_ratio = cont / len(gsm)

print(f"Num. contaminated: {cont}; ratio: {cont_ratio}")