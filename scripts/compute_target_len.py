import os, multiprocessing, datasets
from transformers import AutoTokenizer

### Tulu Format ###
BASE_BOS: str = "<s>"
TURN_SEP: str = "\n"

USER_BOS: str = "<|user|>\n"
USER_EOS: str = ""

EMBED_BOS: str = "\n<|embed|>\n"
# Am embed eos is useless as there is no generative loss on it so it won't be learned
# & it does not add anything new; It only makes sense for lasttoken pooling
EMBED_EOS: str = ""

ASSISTANT_BOS: str = "\n<|assistant|>\n"
ASSISTANT_EOS: str = "</s>"

tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
ds = datasets.load_dataset("json", data_files="/data/niklas/gritlm/e5/tuluv2.jsonl", split="train")

MAX_LEN=2048

num_proc = max(multiprocessing.cpu_count()-2, 1)
ds = ds.filter(
    lambda ex: len(tokenizer.tokenize(USER_BOS + ex["text"][0] + USER_EOS + ASSISTANT_BOS)) < MAX_LEN,
    num_proc=num_proc,
)

"""
g_lens = [
    sum([
        len(tokenizer.tokenize(z.strip() + ASSISTANT_EOS))
        for z in f["text"][1::2]
    ]) for f in ds
]
"""
# Mapped version with ds

def compute_len(ex):
    ex["len"] = sum([
        len(tokenizer.tokenize(z.strip() + ASSISTANT_EOS))
        for z in ex["text"][1::2]
    ])
    return ex

g_lens = ds.map(compute_len, num_proc=num_proc)
g_lens = [ex["len"] for ex in g_lens]

# Compute median / mean
import statistics
print(statistics.median(g_lens))
print(statistics.mean(g_lens))
#149.0
#670.8825593716886

