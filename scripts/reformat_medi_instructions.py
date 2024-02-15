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
            d[k] = [(d[k][0], d[k][1])]
        json.dump(d, g, ensure_ascii=False)
        g.write('\n')
