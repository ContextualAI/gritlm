from datasets import load_dataset


ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

def multiturn(x):
    texts = []
    num_messages = len(x["messages"])
    # If it ends with a user turn, ignore that last one
    cur_user = ""
    for i in range(num_messages):
        if x["messages"][i]["role"] == "system":
            cur_user += x["messages"][i]["content"] + "\n"
        elif x["messages"][i]["role"] == "user":
            texts.append(cur_user + x["messages"][i]["content"])
            cur_user = ""
        elif x["messages"][i]["role"] == "assistant":
            texts.append(x["messages"][i]["content"])
        else: raise ValueError("Unknown role: " + x["messages"][i]["role"])
    return {"text": texts}

ds = ds.map(multiturn, remove_columns=["messages", "prompt", "prompt_id"])
ds.to_json("ultrachat.jsonl", orient="records", force_ascii=False)