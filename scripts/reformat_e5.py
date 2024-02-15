import json

DATA = "/data/niklas/gritlm/e5_train.jsonl"

with open(DATA, "r") as f, open(DATA.replace(".jsonl", "_format.jsonl"), "w") as out:
    for line in f:
        data = json.loads(line)
        assert len(data['query']) == 2  
        assert data['query'][0].startswith("Instruct: ")
        assert data['query'][0].endswith("\nQuery: ")
        data['query'][0] = data['query'][0][len("Instruct: "):-len("\nQuery: ")]
        out.write(
            json.dumps(data, ensure_ascii=False)
            + "\n"
        )
#"""
"""
# With datasets
DATA = "/data/niklas/gritlm/e5_train_ds.jsonl"

DS_TO_SAMPLES = {}

with open(DATA, "r") as f:
    for line in f:
        data = json.loads(line)
        assert len(data['query']) == 2  
        assert data['query'][0].startswith("Instruct: ")
        assert data['query'][0].endswith("\nQuery: ")
        data['query'][0] = data['query'][0][len("Instruct: "):-len("\nQuery: ")]

        DS_TO_SAMPLES.setdefault(data['source'], []).append(data)

for ds, samples in DS_TO_SAMPLES.items():
    with open(
        DATA.replace("gritlm", "gritlm/e5ds").replace(".jsonl", f"_{ds}_format.jsonl"), "w"
    ) as out:
        for data in samples:
            out.write(
                json.dumps(data, ensure_ascii=False)
                + "\n"
            )
"""
"""Source Counter
with open("/data/niklas/gritlm/e5_train_ds.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        source = data['source']
        if source not in DS_TO_SAMPLES:
            DS_TO_SAMPLES[source] = 0
        DS_TO_SAMPLES[source] += 1
"""