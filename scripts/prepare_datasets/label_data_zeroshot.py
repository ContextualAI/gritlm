import json
import random

task = 'zeroshot-train-multikilt'
with open(f"/home2/huggingface/datasets/retrieval/KIB/dpr/v0.20200817/{task}.json") as f:
    data = json.load(f)

query_text_type = ['phrase','words','phrase','words','query']
doc_text_type = ['descriptions','information','introductions','documents','passages','sentences']
skip_count = 0
with open(f"/home2/huggingface/datasets/sentence-transformer-embedding-data/1107/{task}.jsonl",'w') as f:
    for e in data:
        assert isinstance(e['question'],str)
        assert isinstance(e['positive_ctxs'],list)
        assert isinstance(e['positive_ctxs'][0], dict)
        assert isinstance(e['positive_ctxs'][0]['text'], str)
        assert isinstance(e['positive_ctxs'][0]['title'], str)
        assert isinstance(e['hard_negative_ctxs'],list)
        assert isinstance(e['hard_negative_ctxs'][0], dict)
        assert isinstance(e['hard_negative_ctxs'][0]['text'], str)
        cur_doc_type = random.choice(doc_text_type)
        query_components = e['question'].split('[SEP]')
        if e['positive_ctxs'][0]['title'].strip() != query_components[0].strip():
            skip_count += 1
            continue
        assert len(query_components)==2
        cur_dict = {
            'query': query_components[0].strip(),
            'pos': [e['positive_ctxs'][0]['text']],
            'neg': [e['hard_negative_ctxs'][0]['text']],
            'task': task,
            'instruction': {
                'query': f"Represent the {random.choice(query_text_type)} for retrieving the {cur_doc_type} about the {query_components[1].strip()}:",
                'pos': f"Represent the {cur_doc_type} about the {query_components[1].strip()} for retrieval:",
                'neg': f"Represent the {cur_doc_type} about the {query_components[1].strip()} for retrieval:",
            }
        }
        f.write(json.dumps(cur_dict)+'\n')
print(skip_count)