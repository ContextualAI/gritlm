"""
Format:
split   dataset filename        sentence1       sentence2       label
train   SNLI    snli_1.0_train  A person on a horse jumps over a broken down airplane.  A person is training his horse for a competition.        neutral
train   SNLI    snli_1.0_train  A person on a horse jumps over a broken down airplane.  A person is at a diner, ordering an omelette.    contradiction

Turn into:
{"query": str, "pos": List[str], "neg": List[str]}
{"query": str, "pos": List[str], "neg": List[str]}
{"query": str, "pos": List[str], "neg": List[str]}
"""
import csv
import gzip
import json
import random

nli_dataset_path = 'allnli.tsv.gz'
def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)

train_data = {}
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            sent1 = row['sentence1'].strip()
            sent2 = row['sentence2'].strip()

            add_to_samples(sent1, sent2, row['label'])
            add_to_samples(sent2, sent1, row['label'])  #Also add the opposite

with open('allnli.jsonl', 'w') as fOut:
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            others['entailment'] = list(others['entailment'])
            others['contradiction'] = list(others['contradiction'])
            json.dump({
                'query': sent1, 
                'pos': [random.choice(others['entailment'])],
                'neg': [random.choice(others['contradiction'])]}, 
            fOut)
            fOut.write('\n')
            json.dump({
                'query': random.choice(others['entailment']), 
                'pos': [sent1],
                'neg': [random.choice(others['contradiction'])]}, 
            fOut)
            fOut.write('\n')
