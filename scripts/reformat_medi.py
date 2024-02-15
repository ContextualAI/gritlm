import itertools
import json

DEFAULT_INSTRUCTION = "Represent this text:"

# Turn from json to list of dict (jsonl) for each task
with open('/data/niklas/gritlm/medi-data-hardnegatives.json', 'r') as f:
    # Sort by task_id
    data = json.load(f)
    for task_id, dg in itertools.groupby(data, key=lambda x: x["task_id"]):
        print("Running:", task_id)
        with open(f'/data/niklas/gritlm/meditaskdata/medi-data-hardnegatives-instruct-{task_id}.jsonl', 'w') as g:
            for d in dg:
                for k in ["query", "pos", "neg"]:
                    assert len(d[k]) == 2
                    # Always have an instruction
                    if not d[k][0]:
                        d[k][0] = DEFAULT_INSTRUCTION
                    if k == "query":
                        d[k] = [d[k][0], d[k][1]]
                    else:
                        d[k] = [[d[k][0], d[k][1]]]
                json.dump(d, g, ensure_ascii=False)
                g.write('\n')
"""Slow version (-> same as above but slower)
with open('medi-data-hardnegatives.json', 'r') as f:
    data = json.load(f)
    for d in data:
        for k in ["query", "pos", "neg"]:
            assert len(d[k]) == 2
            # Always have an instruction
            if not d[k][0]:
                d[k][0] = DEFAULT_INSTRUCTION
            if k == "query":
                d[k] = [d[k][0], d[k][1]]
            else:
                d[k] = [[d[k][0], d[k][1]]]
        with open(f'medi-data-hardnegatives-instruct-{d["task_id"]}.jsonl', 'w') as g:
            json.dump(d, g, ensure_ascii=False)
            g.write('\n')
"""
"""Version without separate files (-> in-batch items may come from different tasks with this)
import json

DEFAULT_INSTRUCTION = "Represent this text:"

# Turn from json to list of dict (jsonl)
with open('medi-data-hardnegatives.json', 'r') as f, open('medi-data-hardnegatives-instruct.jsonl', 'w') as g:
    data = json.load(f)
    for d in data:
        for k in ["query", "pos", "neg"]:
            assert len(d[k]) == 2
            # Always have an instruction
            if not d[k][0]:
                d[k][0] = DEFAULT_INSTRUCTION
            if k == "query":
                d[k] = [d[k][0], d[k][1]]
            else:
                d[k] = [[d[k][0], d[k][1]]]
        json.dump(d, g, ensure_ascii=False)
        g.write('\n')
"""
"""Version without separating out instructions  (-> instruction is part of the embedding with this)
import json

separator = " "

# Turn from json to list of dict (jsonl)
with open('medi-data.json', 'r') as f, open('medi-data.jsonl', 'w') as g:
    data = json.load(f)
    for d in data:
        for k in ["query", "pos", "neg"]:
            assert len(d[k]) == 2
            d[k] = separator.join(d[k])
        json.dump(d, g, ensure_ascii=False)
        g.write('\n')
"""
