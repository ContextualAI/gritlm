import json
import random

task = 'nq-train-multikilt'
with open(f"/home2/huggingface/datasets/retrieval/KIB/dpr/v0.20200817/{task}.json") as f:
    data = json.load(f)
print(len(data))

query_text_type = ['question','query','post']
doc_text_type = ['passage','paragraph','document','information']
with open(f"/home2/huggingface/datasets/sentence-transformer-embedding-data/1107/{task}.jsonl",'w') as f:
    for idx,e in enumerate(data):
        if idx >= 200000:
            break
        assert isinstance(e['question'],str)
        assert isinstance(e['positive_ctxs'],list)
        assert isinstance(e['positive_ctxs'][0], dict)
        assert isinstance(e['positive_ctxs'][0]['text'], str)
        assert isinstance(e['hard_negative_ctxs'],list)
        assert isinstance(e['hard_negative_ctxs'][0], dict)
        assert isinstance(e['hard_negative_ctxs'][0]['text'], str)
        cur_doc_type = random.choice(doc_text_type)
        if cur_doc_type != 'information':
            query_doc_type = cur_doc_type + 's'
        else:
            query_doc_type = cur_doc_type
        if random.random()>0.5:
            added_domain = ''
        else:
            added_domain = 'Wikipedia '
        cur_dict = {
            'query': e['question'].strip(),
            'pos': [e['positive_ctxs'][0]['text']],
            'neg': [e['hard_negative_ctxs'][0]['text']],
            'task': task,
            'instruction': {
                'query': f"Represent the {random.choice(query_text_type)} for retrieving relevant {added_domain}{query_doc_type}:",
                'pos': f"Represent the {added_domain}{cur_doc_type} for retrieval:",
                'neg': f"Represent the {added_domain}{cur_doc_type} for retrieval:",
            }
        }
        f.write(json.dumps(cur_dict)+'\n')
