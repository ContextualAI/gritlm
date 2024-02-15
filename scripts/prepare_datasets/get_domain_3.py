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
    #     'task_name': 'WikiAnswers',
    #     'task_num': 1000000
    # },
    # {
    #     'task_name': 'flickr30k_captions',
    #     'task_num': 31783
    # },
    {
        'task_name': 'coco_captions',
        'task_num': 82783
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
            assert isinstance(instance['set'],list) and isinstance(instance['set'][0],str)
            queries.append({
                'query': instance['set'][0],
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



        

