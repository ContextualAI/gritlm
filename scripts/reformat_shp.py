from datasets import load_dataset

ds = load_dataset("stanfordnlp/SHP-2", split="train")

def reformat(x):
    texts = [x['history']]
    if x['labels'] == 1:
        texts.append(x['human_ref_A'])
    elif x['labels'] == 0:
        texts.append(x['human_ref_B'])
    else:
        raise ValueError("Unknown label: " + str(x['labels']))
    return {"text": texts}

ds = ds.map(reformat, remove_columns=ds.column_names)
ds.to_json("shp2full.jsonl", orient="records", force_ascii=False)

