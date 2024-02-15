from datasets import load_dataset

ds = load_dataset("bigcode/oasst-octopack", split="train")

# Optionally filter for english
ds = ds.filter(lambda x: x["lang"] == "en")

# Only single-turn
"""
ds = ds.map(
    lambda x: {"text": x["conversations"][0]['text'], x["conversations"][1]['text']},
    remove_columns=["conversations", "lang"],
)
"""

# Any number of turns
def multiturn(x):
    texts = []
    turns = len(x["conversations"])
    # If it ends with a user turn, ignore that last one
    turns = turns // 2
    for i in range(turns):
        texts.append(x["conversations"][i*2]["text"])
        texts.append(x["conversations"][i*2+1]["text"])
    return {"text": texts}

ds = ds.map(multiturn, remove_columns=["conversations", "lang"])
ds.to_json("oasst_octopack_en.jsonl", orient="records", force_ascii=False)