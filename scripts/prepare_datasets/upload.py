import os
from datasets import load_dataset

# tasks = ['WikiAnswers', 'reddit-title-body', 'medmcqa', 'SimpleWiki', 'gooaq_pairs', 'squad_pairs', 'searchQA_top5_snippets', 'pubmedqa', 'npr', 'S2ORC_title_abstract', 'wikihow', 'trex-train-multikilt', 'nq-train-multikilt', 'gigaword', 'yahoo_answers_title_answer', 'fever-train-multikilt', 'flickr30k_captions', 'sentence-compression', 'zeroshot-train-multikilt', 'PAQ_pairs', 'hotpotqa-train-multikilt', 'scitldr', 'xsum']
tasks = ['codesearchnet']
# 'codesearchnet',
for task in tasks:
    print('current uploading:',task)
    all_data = load_dataset('json', data_files=f"/home2/huggingface/datasets/sentence-transformer-embedding-data/1107/{task}.jsonl",
                            cache_dir=f'/home2/huggingface/datasets/emb_train_1107/{task}')
    all_data.push_to_hub(f'multi-train/{task}_1107',token='YOUR_HF_KEY')



