import os

import datasets

data = "/data/niklas/gritlm/medituluv2"

data_files = [os.path.join(data, x) for x in os.listdir(data)] if os.path.isdir(data) else [data]

def check_tokens(dataset):
    def filter_fn(example):
        if len(example["query"][1].strip()) == 0:
            print(f"Empty txt: {example}")
        for ex in example["pos"] + example["neg"]:
            if len(ex[1].strip()) == 0:
                print(f"Empty txt: {example}")
        return True
    return dataset.filter(filter_fn)

for file in data_files:
    tmp_ds = datasets.load_dataset('json', data_files=file, split='train')
    if "query" in tmp_ds.features:
        if 'query' in tmp_ds[0]:
            # - 8 to leave buffer for special tokens
            tmp_ds = check_tokens(tmp_ds)
        else: 
            raise NotImplementedError