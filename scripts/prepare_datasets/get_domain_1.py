import os
import random
import json
import multiprocessing as mp
from tqdm import tqdm
from tqdm import trange
from util import worker

# task = 'npr'

numbers = [
    # {
    #     'task_name': 'gooaq_pairs',
    #     'task_num': 1000000,
    # },
    # {
    #     'task_name': 'yahoo_answers_title_answer',
    #     'task_num': 1000000
    # },
    # {
    #     'task_name': 'stackexchange_duplicate_questions_title_title',
    #     'task_num': 304525
    # },
    # {
    #     'task_name': 'eli5_question_answer',
    #     'task_num': 325474
    # },
    # {
    #     'task_name': 'squad_pairs',
    #     'task_num': 87599
    # },
    # {
    #     'task_name': 'wikihow',
    #     'task_num': 128542
    # },
    # {
    #     'task_name': 'amazon_review_2018',
    #     'task_num': 1000000
    # },
    # {
    #     'task_name': 'S2ORC_title_abstract',
    #     'task_num': 1000000
    # },
    # # {
    # #     'task_name': 'searchQA_top5_snippets',
    # #     'task_num': 117220
    # # },
    {
        'task_name': 'agnews',
        'task_num': 1000000
    },
    {
        'task_name': 'npr',
        'task_num': 594383,
    },
    {
        'task_name': 'SimpleWiki',
        'task_num': 102225
    },
    {
        'task_name': 'PAQ_pairs',
        'task_num': 1000000
    },
    {
        'task_name': 'altlex',
        'task_num': 102225
    },
    {
        'task_name': 'ccnews_title_text',
        'task_num': 614664
    },
    {
        'task_name': 'xsum',
        'task_num': 226711
    },
    {
        'task_name': 'codesearchnet',
        'task_num': 1151413
    },
    {
        'task_name': 'sentence-compression',
        'task_num': 180000
    },
    {
        'task_name': 'cnn_dailymail',
        'task_num': 311971
    },


]

# num = None
# for item in numbers:
#     if item['task_name']==task:
#         num = item['task_num']
# assert num is not None


for item in numbers:
    num = item['task_num']
    task = item['task_name']
    print(task)
    queries = []
    if not os.path.isdir('domains'):
        os.makedirs('domains',exist_ok=True)
    with open(f'/home2/huggingface/datasets/sentence-transformer-embedding-data/{task}.jsonl') as f:
        for i in trange(num):
            line = f.readline()
            instance = json.loads(line)
            queries.append({
                'query': instance[0],
                'idx': i
            })
    written_indices = {}
    for file in os.listdir('domains'):
        if file.endswith('.json') and file.startswith(f'{task}_'):
            written_indices[int(file[len(f'{task}_'):-len('.json')])] = True
    print('skip',len(written_indices),'indices')
    new_inference_args = []
    for a in queries:
        if not a['idx'] in written_indices:
            new_inference_args.append(a)
    inference_args = new_inference_args
    print(len(inference_args))
    inference_args = random.sample(inference_args,1000)
    with mp.Pool(32) as pool, tqdm(total=len(inference_args), desc='inference') as pbar:
        for return_contents in pool.imap_unordered(worker, inference_args):
            pbar.update()
            if return_contents is not None and 'query' in return_contents and 'domain' in return_contents:
                with open(os.path.join('domains',f"{task}_{return_contents['idx']}.json"),'w') as f:
                    json.dump(return_contents,f,indent=2)



        

