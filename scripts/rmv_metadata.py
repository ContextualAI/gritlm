"""
Remove metadata from jsonl
{"_id": "0", "text": "0-dimensional biomaterials lack inductive properties.", "metadata": {}}
becomes
{"_id": "0", "text": "0-dimensional biomaterials lack inductive properties."}
"""
import json
import os

PATHS = [
    "arguana",
    "dbpedia",
    "msmarco",
    "msmarco-v2",
    "scidocs",
    "fiqa",
    "fever",
    "nq",
    "hotpotqa",
    "touche2020",
    "trec-covid",
    "quora",
    "climate-fever",
    "nfcorpus",
]

for name in PATHS:
    for s in ["queries.jsonl", "corpus.jsonl"]:
        print(f"Processing {name}/{s}")
        path = os.path.join(name, s)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if "metadata" not in lines[0]:
                continue

        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                data = json.loads(line)
                del data["metadata"]
                f.write(json.dumps(data) + "\n")

