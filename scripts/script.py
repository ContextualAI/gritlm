import json
import os
import torch
import random
import argparse
from datasets import load_dataset
from tqdm import trange
from transformers import AutoTokenizer,pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--max_examples", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_generation_length", type=int, default=300)
parser.add_argument("--model_max_length", type=int, default=2048)
parser.add_argument("--output_dir", type=str, default='./')
parser.add_argument("--prompt_examples_dir", type=str, default='prompt_examples')
parser.add_argument("--data_id", type=str, default='multi-train/emb_train_1105_qqp')
args = parser.parse_args()

maximum = args.max_examples
output_dir = args.output_dir
data_id = args.data_id
batch_size = args.batch_size
prompt_examples_dir = args.prompt_examples_dir
max_generation_length = args.max_generation_length
model_max_length = args.model_max_length

def cap(d,m):
    if len(d)<m:
        return d
    if m is not None:
        indices = list(range(len(d)))
        random.shuffle(indices)
        indices = indices[:m]
        d = d.select(indices)
    return d

def format_prompt(example,cur_query,cur_doc):
    return f'''<|system|>
You are given a query and a document that should be matched by embeddings. Based on the two please write a clear instruction in one sentence about how the query should be embedded to match the document.

For example:
Query: {example["query"].strip()}
Document: {example["doc"].strip()}

Target intent: {example["intent"].strip()}
Target domain: {example["domain"].strip()}
Target unit: {example["unit"].strip()}
Instruction: {example["instruction"].strip()}
<|user|>
Please write the target intent, target domain, target unit and instruction for the following query to match the document.
The instruction should always start with Represent.

Query: {cur_query.strip()}
Document: {cur_doc.strip()}</s>
<|assistant|>
Target intent: '''

def process_query_doc(tokenizer,query,doc,remain_length):
    query_tok = tokenizer(query)['input_ids']
    doc_tok = tokenizer(doc)['input_ids']
    if len(query_tok)+len(doc_tok)<remain_length:
        return query,doc
    if len(doc_tok)/len(query_tok)>5 and 2*len(query_tok)<remain_length:
        cut_doc_ids = doc_tok[1:remain_length-len(query_tok)]
        processed_doc = tokenizer.decode(cut_doc_ids)
        processed_query = query
    elif len(query_tok)/len(doc_tok)>5 and 2*len(doc_tok)<remain_length:
        cut_query_ids = query_tok[1:remain_length-len(doc_tok)]
        processed_query = tokenizer.decode(cut_query_ids)
        processed_doc = doc
    else:
        query_length = len(query_tok)
        doc_length = len(doc_tok)
        if query_length>doc_length:
            processed_doc_length = max(int(remain_length*doc_length/(query_length+doc_length)),50)
            processed_query_length = remain_length-doc_length
        else:
            processed_query_length = max(int(remain_length*query_tok/(query_length+doc_length)),50)
            processed_doc_length = remain_length-query_length
        processed_query = query
        processed_doc = doc
        if processed_query_length<query_length:
            cut_query_ids = query_tok[1:processed_query_length+1]
            processed_query = tokenizer.decode(cut_query_ids)
        if processed_doc_length<doc_length:
            cut_doc_ids = doc_tok[1:processed_doc_length+1]
            processed_doc = tokenizer.decode(cut_doc_ids)
    assert query.startswith(processed_query) and doc.startswith(processed_doc)
    return processed_query,processed_doc

full_data = load_dataset(data_id,token='YOUR_HF_KEY')['train']
cur_dataset = cap(full_data,maximum)
total_num = len(cur_dataset)
model = 'HuggingFaceH4/zephyr-7b-beta'
tokenizer = AutoTokenizer.from_pretrained(model, truncation_side="right")
inference_pipeline = pipeline("text-generation",model=model,torch_dtype=torch.float16,device_map="auto")
inference_pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id
task = data_id.split('/')[-1]
with open(os.path.join(prompt_examples_dir,f"{task}.json")) as f:
    written_example = json.load(f)
if not os.path.isdir(os.path.join(output_dir,task)):
    os.makedirs(os.path.join(output_dir,task),exist_ok=True)
base_prompt = f'''<|system|>
You are given a query and a document that should be matched by embeddings. Based on the two please write a clear instruction in one sentence about how the query should be embedded to match the document.

For example:
Query: {written_example["query"].strip()}
Document: {written_example["doc"].strip()}

Target intent: {written_example["intent"].strip()}
Target domain: {written_example["domain"].strip()}
Target unit: {written_example["unit"].strip()}
Instruction: {written_example["instruction"].strip()}
<|user|>
Please write the target intent, target domain, target unit and instruction for the following query to match the document.
The instruction should always start with Represent.

Query: 
Document: </s>
<|assistant|>
Target intent: '''
remaining_length = model_max_length-max_generation_length-len(tokenizer(base_prompt)['input_ids'])
skip_count = 0
for i in trange(0,total_num,batch_size):
    try:
        batch_prompts = []
        cur_batch = [cur_dataset[j] for j in range(i, min(i + batch_size, total_num))]
        for e in cur_batch:
            processed_query, processed_doc = process_query_doc(
                tokenizer=tokenizer,
                query=e['query'],
                doc=e['pos'][0],
                remain_length=remaining_length
            )
            batch_prompts.append(format_prompt(example=written_example,cur_query=processed_query,cur_doc=processed_doc))
        raw_results = inference_pipeline(batch_prompts, do_sample=False, num_return_sequences=1,
                               eos_token_id=tokenizer.eos_token_id, max_length=2048, batch_size=batch_size)
        assert len(raw_results)==len(cur_batch),f"{len(raw_results)},\n {len(cur_batch)},\n{raw_results}"
        assert len(cur_batch)==len(batch_prompts)
        count = -1
        for e,r,p in zip(cur_batch,raw_results,batch_prompts):
            count += 1
            lines = ('Target intent: '+r[0]['generated_text'][len(p):]).split('\n')
            intent = None
            domain = None
            unit = None
            instruction = None
            for l in lines:
                if l.lower().startswith('target intent:'):
                    intent = l.split(':')[1]
                elif l.lower().startswith('target domain:'):
                    domain = l.split(':')[1]
                elif l.lower().startswith('target unit:'):
                    unit = l.split(':')[1]
                elif l.lower().startswith('instruction:'):
                    instruction = l.split(':')[1]
                if intent is not None and domain is not None and unit is not None and instruction is not None:
                    break
            assert intent is not None and domain is not None and unit is not None and instruction is not None
            cur_dict = {
                'query': e['query'].strip(),
                'doc': e['pos'][0].strip(),
                'intent': intent.strip(),
                'domain': domain.strip(),
                'unit': unit.strip(),
                'instruction': instruction.strip()
            }
            with open(os.path.join(output_dir,task,f"{i+count}.json"),'w') as f:
                json.dump(cur_dict,f,indent=2)
    except Exception as e:
        skip_count += 1
        continue
print(skip_count*batch_size,'are not generated')
