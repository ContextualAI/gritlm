from datasets import load_dataset

data_keys = ['multi-train/emb_train_1105_qqp','multi-train/emb_train_1105_stackexchange',
                   'multi-train/emb_train_1105_specter','multi-train/emb_train_1105_AllNLI',
                   'multi-train/emb_train_1105_msmarco']
for k in data_keys:
    d = load_dataset(k,token='YOUR_TOKEN',
                                        cache_dir='/home2/huggingface/datasets/v1105')['train']
    print(d)

