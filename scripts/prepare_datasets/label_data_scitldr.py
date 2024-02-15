import os
import json
import torch
import random
import warnings
import argparse
import numpy as np
from tqdm import trange
from collections import defaultdict
from datasets import load_dataset
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoModel,AutoTokenizer
from util import calculate_llama2_embedding


def compute_similarity(q_reps, p_reps):
    if len(p_reps.size()) == 2: return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))

parser = argparse.ArgumentParser()
warnings.filterwarnings("ignore")
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

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

retrieval_model = INSTRUCTOR('hkunlp/instructor-large')

queries = []
docs = []
doc_indices = defaultdict(list)
global_count = 0
full_data = load_dataset('Blaise-g/scitldr',cache_dir='/home2/huggingface/datasets/',split='train')
for instance in full_data:
    assert isinstance(instance['source'], str)
    assert isinstance(instance['target'], str)
    queries.append(instance['source'])
    docs.append(instance['target'])
    doc_indices[instance['target']].append(global_count)
    global_count += 1
    assert len(queries)==global_count
    assert len(docs) == global_count
print(global_count)
num = global_count

instructor_score_violate_count = 0
if not os.path.isfile(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_mined_negatives.jsonl'):
    model_llama2 = AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.float16,
                                             token='YOUR_TOKEN').cuda()
    tokenizer_llama2 = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False,model_max_length=2048,padding_side="right")
    tokenizer_llama2.pad_token = tokenizer_llama2.unk_token
    queries_inst = [[query_instruction,q] for q in queries]
    docs_inst = [[doc_instruction,d] for d in docs]
    negative_map = defaultdict(list)
    if os.path.isfile(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_llama2_emb.pt'):
        print('use cached llama2 embeddings')
        doc_emb_llama2 = torch.load(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_llama2_emb.pt')
    else:
        print('recalculate llama2 embeddings')
        doc_emb_llama2 = calculate_llama2_embedding(model=model_llama2, tokenizer=tokenizer_llama2,
                                                    texts=docs,
                                                    batch_size=4).cpu()
        doc_emb_llama2 = torch.nn.functional.normalize(doc_emb_llama2, dim=-1)
        torch.save(doc_emb_llama2, f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_llama2_emb.pt')
    if os.path.isfile(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_instructor_emb.pt'):
        print('use cached instructor embeddings')
        doc_emb_instructor = torch.load(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_instructor_emb.pt')
    else:
        doc_emb_instructor = retrieval_model.encode(docs_inst, show_progress_bar=True,
                                                    batch_size=128)
        doc_emb_instructor = torch.nn.functional.normalize(torch.from_numpy(doc_emb_instructor), dim=-1)
        torch.save(doc_emb_instructor,f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_instructor_emb.pt')
    assert len(doc_emb_instructor) == len(doc_emb_llama2)
    assert len(doc_emb_instructor) == num
    for start_idx in range(0,num,query_batch_size):
        query_emb_instructor = retrieval_model.encode(queries_inst[start_idx:start_idx+query_batch_size], show_progress_bar=True, batch_size=128)
        query_emb_llama2 = calculate_llama2_embedding(model=model_llama2, tokenizer=tokenizer_llama2, texts=queries[start_idx:start_idx+query_batch_size],
                                                      batch_size=4).cpu()
        query_emb_instructor = torch.nn.functional.normalize(torch.from_numpy(query_emb_instructor), dim=-1)
        query_emb_llama2 = torch.nn.functional.normalize(query_emb_llama2, dim=-1)
        assert len(query_emb_llama2) == len(query_emb_instructor)

        instructor_scores = compute_similarity(query_emb_instructor,doc_emb_instructor)
        instructor_scores = (instructor_scores>0.9).long()*100
        llama2_scores = compute_similarity(query_emb_llama2,doc_emb_llama2)
        llama2_scores -= instructor_scores
        print('llama2_scores shape', llama2_scores.shape,start_idx)
        cur_queries = queries[start_idx:start_idx + query_batch_size]
        assert len(cur_queries)==len(llama2_scores),f"{len(cur_queries)}, {len(llama2_scores)}"
        assert len(docs)==len(llama2_scores[0]),f"{len(docs)}, {len(llama2_scores[0])}"
        score_idx = 0
        for q,s in zip(cur_queries,llama2_scores):
            assert queries[start_idx+score_idx]==q
            cur_doc = docs[start_idx+score_idx]
            for remove_idx in doc_indices[cur_doc]:
                s[remove_idx] -= 100
            selected_idx = np.argmax(s)
            if instructor_scores[score_idx][selected_idx]>0.9:
                instructor_score_violate_count += 1
                print('instructor score larger than 0.9')
            score_idx += 1
            negative_map[q].append(docs[selected_idx])
            assert cur_doc!=negative_map[q]
        print('length of negative map:', len(negative_map),start_idx)
    with open(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_mined_negatives.jsonl','w') as f:
        for query1,negative1 in negative_map.items():
            assert isinstance(query1,str) and isinstance(negative1,list) and isinstance(negative1[0],str)
            f.write(json.dumps([query1,negative1])+'\n')
else:
    print('loading cached negatives')
    negative_map = {}
    with open(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}_mined_negatives.jsonl') as f:
        for iter_idx in trange(num):
            cur_line = f.readline()
            cur_query_negative = json.loads(cur_line)
            negative_map[cur_query_negative[0]] = cur_query_negative[1]

queries_to_write = []
positives_to_write = []
negatives_to_write = []
instructions_to_write = []
empty_domain_count = 0
assert len(queries)==len(docs),f"{len(queries)}, {len(docs)}"

negative_not_found = 0
for q,d in zip(queries,docs):
    assert isinstance(q, str) and isinstance(d, str)
    cur_negative_doc = None
    assert isinstance(negative_map[q],list) and isinstance(negative_map[q][0],str)
    for doc_iter in negative_map[q]:
        if doc_iter!=d:
            cur_negative_doc = doc_iter
            negatives_to_write.append(doc_iter)
            break
    if cur_negative_doc is None:
        negative_not_found += 1
        continue
    queries_to_write.append(q)
    positives_to_write.append(d)

    doc_text_type = random.choice(doc_text_types)
    instructions_to_write.append({
        'query': f"Represent the {default_domain}{random.choice(query_text_types)}:",
        'pos': f"Represent the {default_domain}{doc_text_type}:",
        'neg': f"Represent the {default_domain}{doc_text_type}:",
    })

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
print('instructor_score_violate_count:',instructor_score_violate_count)
print('negative_not_found:',negative_not_found)