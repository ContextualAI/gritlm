import os
import json
import torch
import random
import warnings
import argparse
import numpy as np
from tqdm import trange
from collections import defaultdict
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoModel,AutoTokenizer
from util import calculate_llama2_embedding

parser = argparse.ArgumentParser()
warnings.filterwarnings("ignore")
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

with open(args.config) as f:
    config = json.load(f)
task = config['task']
num = config['num']
query_batch_size = config['query_batch_size']
force_recreate = config['force_recreate']
query_instruction = config['query_instruction']
doc_instruction = config['doc_instruction']
default_domain = config['default_domain']
query_text_types = config['query_text_types']
doc_text_types = config['doc_text_types']
cur_default_domain_coexist = config['cur_default_domain_coexist']
llama2_batch_size = config['llama2_batch_size']

retrieval_model = INSTRUCTOR('hkunlp/instructor-large')

query_domain = {}
labeled_queries = []
for file in os.listdir('domains'):
    if file.endswith('.json') and file.startswith(f'{task}_'):
        with open(os.path.join('domains',file)) as f:
            cur_dict = json.load(f)
        query_domain[cur_dict['query']] = cur_dict['domain']
        labeled_queries.append(cur_dict['query'])
print('labeled queries:',len(labeled_queries))

queries = []
docs = []
doc_indices = defaultdict(list)
with open(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}.jsonl') as f:
    for i in trange(num):
        line = f.readline()
        instance = json.loads(line)
        assert len(instance) == 2
        assert isinstance(instance[0], str)
        assert isinstance(instance[1], str)
        queries.append(instance[0])
        docs.append(instance[1])
        doc_indices[instance[1]].append(i)
if not os.path.isfile(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_mined_negatives.jsonl') or force_recreate:
    model_llama2 = AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.float16,
                                             token='YOUR_HF_KEY').cuda()
    tokenizer_llama2 = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False)
    tokenizer_llama2.pad_token = tokenizer_llama2.unk_token
    queries_inst = [[query_instruction,q] for q in queries]
    docs_inst = [[doc_instruction,d] for d in docs]
    negative_map = {}
    doc_emb_instructor = retrieval_model.encode(docs_inst, show_progress_bar=True,batch_size=128)
    doc_emb_instructor = torch.from_numpy(doc_emb_instructor)
    doc_emb_llama2 = calculate_llama2_embedding(model=model_llama2, tokenizer=tokenizer_llama2,
                                                texts=docs,
                                                batch_size=llama2_batch_size).cpu()
    assert len(doc_emb_instructor) == len(doc_emb_llama2)
    assert len(doc_emb_instructor) == num
    for start_idx in range(0,num,query_batch_size):
        query_emb_instructor = retrieval_model.encode(queries_inst[start_idx:start_idx+query_batch_size], show_progress_bar=True, batch_size=128)
        query_emb_instructor = torch.from_numpy(query_emb_instructor)
        query_emb_llama2 = calculate_llama2_embedding(model=model_llama2, tokenizer=tokenizer_llama2, texts=queries[start_idx:start_idx+query_batch_size],
                                                      batch_size=4).cpu()
        assert len(query_emb_llama2) == len(query_emb_instructor)
        cur_queries = queries[start_idx:start_idx + query_batch_size]
        score_idx = 0
        for q,instructor_query_emb,llama2_query_emb in zip(cur_queries,query_emb_instructor,query_emb_llama2):
            instructor_scores = cos(doc_emb_instructor, instructor_query_emb)
            # print('instructor_scores shape',instructor_scores.shape)
            llama2_scores = cos(doc_emb_llama2, llama2_query_emb)
            # print('llama2_scores shape',llama2_scores.shape)
            assert len(docs) == len(llama2_scores), f"{len(docs)}, {len(llama2_scores)}"
            assert len(docs) == len(instructor_scores), f"{len(docs)}, {len(instructor_scores)}"
            instructor_scores = (instructor_scores > 0.9).long() * 100
            llama2_scores -= instructor_scores
            s = llama2_scores
            assert queries[start_idx+score_idx]==q
            cur_doc = docs[start_idx+score_idx]
            for remove_idx in doc_indices[cur_doc]:
                s[remove_idx] -= 100
            selected_idx = np.argmax(s)
            assert instructor_scores[selected_idx]<=0.9
            score_idx += 1
            negative_map[q] = docs[selected_idx]
            assert cur_doc!=negative_map[q]
        print('length of negative map:', len(negative_map),start_idx)
    with open(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_mined_negatives.jsonl','w') as f:
        for query1,negative1 in negative_map.items():
            assert isinstance(query1,str) and isinstance(negative1,str)
            f.write(json.dumps([query1,negative1])+'\n')
else:
    print('loading cached negatives')
    negative_map = {}
    with open(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_mined_negatives.jsonl') as f:
        for iter_idx in trange(num):
            cur_line = f.readline()
            cur_query_negative = json.loads(cur_line)
            negative_map[cur_query_negative[0]] = negative_map[cur_query_negative[1]]

general_words = ['trivia','unclear', 'unknown', 'general', 'ambiguous', 'incomplete', 'nonsensical', 'undefined', 'unspecified', 'uncategorized', 'not specified', 'not clear', 'not known', 'not complete', 'not defined', 'not categorized', 'vague', 'inconclusive', 'obscure', 'enigmatic', 'uncertain', 'murky', 'indefinite', 'nebulous', 'elusive', 'imprecise', 'dubious', 'indeterminate', 'unclarified', 'cryptic', 'equivocal', 'unestablished', 'unconfirmed', 'undiscovered', 'unexplained', 'unresolved', 'unintelligible', 'unspecific', 'unclassified', 'muddled', 'convoluted', 'mysterious', 'unsettled', 'indistinct', 'arcane', 'uncharted', 'non-specific', 'puzzling', 'indeterminable', 'undetermined', 'ill-defined', 'hazy', 'fuzzy', 'opaque', 'clouded', 'mystifying', 'obscured', 'questionable', 'shadowy', 'misleading', 'enigmatical', 'unfathomable', 'tenuous', 'blurry', 'invalid', 'moot', 'cloudy', 'confused', 'undecided', 'inexact', 'unformulated', 'confusing', 'inexplicit', 'indecisive', 'contradictory', 'perplexing', 'doubtful', 'confounding', 'blurred', 'foggy', 'incalculable', 'mistaken', 'illegible', 'unaccounted', 'indecipherable', 'illogical', 'disjointed', 'indescribable', 'impalpable']

gold_queries = [[query_instruction,q] for q in labeled_queries]
gold_query_emb = retrieval_model.encode(gold_queries,show_progress_bar=True,batch_size=128)
gold_query_emb = torch.from_numpy(gold_query_emb)

assert isinstance(queries[0],str) and isinstance(queries,list)
queries_to_label = [[query_instruction,q] for q in queries]
queries_to_label_emb = retrieval_model.encode(queries_to_label,show_progress_bar=True,batch_size=512)
queries_to_label_emb = torch.from_numpy(queries_to_label_emb)
domain_map_query = {}
domain_map_doc = {}
for i in trange(num):
    if queries_to_label[i][1] in query_domain:
        cur_domain = query_domain[queries_to_label[i][1]]
    else:
        score = cos(gold_query_emb, queries_to_label_emb[i])
        # print(score.shape)
        selected_idx = np.argmax(score)
        if score[selected_idx]>0.9:
            cur_domain = query_domain[gold_queries[selected_idx][1]]
        else:
            cur_domain = ''
    domain_map_query[queries_to_label[i][1]] = cur_domain
    domain_map_doc[docs[i]] = cur_domain

queries_to_write = []
positives_to_write = []
negatives_to_write = []
instructions_to_write = []
empty_domain_count = 0
assert len(queries)==len(docs),f"{len(queries)}, {len(docs)}"

def process_domain(cur_domain):
    if cur_domain=='':
        return ''
    if cur_domain.lower().strip()==default_domain.lower().strip():
        return ''
    for w in general_words:
        if w in cur_domain.lower().strip():
            return ''
    return f' about {cur_domain}'

for q,d in zip(queries,docs):
    assert isinstance(q, str) and isinstance(d, str)
    queries_to_write.append(q)
    positives_to_write.append(d)
    negatives_to_write.append(negative_map[q])
    assert d!=negative_map[q]

    cur_query_domain = process_domain(domain_map_query[q])
    cur_pos_domain = process_domain(domain_map_doc[d])
    cur_neg_domain = process_domain(domain_map_doc[negative_map[q]])

    if random.random()>0.5:
        added_domain = ''
    else:
        added_domain = default_domain

    doc_text_type = random.choice(doc_text_types)
    instructions_to_write.append({
        'query': f"Represent the {added_domain}{random.choice(query_text_types)}{cur_query_domain}:",
        'pos': f"Represent the {added_domain}{doc_text_type}{cur_pos_domain}:",
        'neg': f"Represent the {added_domain}{doc_text_type}{cur_neg_domain}:",
    })
    if cur_query_domain=='' and cur_pos_domain=='' and cur_neg_domain=='':
        empty_domain_count += 1

assert len(queries_to_write)==len(positives_to_write)
assert len(queries_to_write)==len(negatives_to_write)
print('loading successful')
print('empty domain count',empty_domain_count)
if not os.path.isdir('/home2/huggingface/datasets/sentence-transformer-embedding-data/1107'):
    os.makedirs('/home2/huggingface/datasets/sentence-transformer-embedding-data/1107',exist_ok=True)
with open(f"/home2/huggingface/datasets/sentence-transformer-embedding-data/1107/{task}.jsonl",'w') as f:
    for q,p,n,instruction in zip(queries_to_write,positives_to_write,negatives_to_write,instructions_to_write):
        assert isinstance(q,str) and isinstance(p,str) and isinstance(n,str)
        assert isinstance(instruction,dict) and 'query' in instruction and 'pos' in instruction and 'neg' in instruction
        f.write(json.dumps({'query': q,'pos': [p],'neg': [n],'task':task,'instruction':instruction})+'\n')



