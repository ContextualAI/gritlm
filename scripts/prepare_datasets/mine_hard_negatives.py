import os
import json
import random
import shutil
import torch
import argparse
import numpy as np
from tqdm import trange
from datasets import load_dataset
from transformers import AutoModel,AutoTokenizer
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

URLS = {
'specter_train_triples': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/specter_train_triples.jsonl.gz',
'AllNLI': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/AllNLI.jsonl.gz',
'msmarco-triplets': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz',
'codesearchnet': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/codesearchnet.jsonl.gz',
'WikiAnswers': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/WikiAnswers.jsonl.gz',
'altlex': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/altlex.jsonl.gz',
'SimpleWiki': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/SimpleWiki.jsonl.gz',
'amazon-qa': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/amazon-qa.jsonl.gz',
'eli5_question_answer': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/eli5_question_answer.jsonl.gz',
'gooaq_pairs': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/gooaq_pairs.jsonl.gz',
'squad_pairs': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/squad_pairs.jsonl.gz',
'searchQA_top5_snippets': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/searchQA_top5_snippets.jsonl.gz',
'coco_captions': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/coco_captions.jsonl.gz',
'ccnews_title_text': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/ccnews_title_text.jsonl.gz',
'npr': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/npr.jsonl.gz',
'wikihow': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/wikihow.jsonl.gz',
'cnn_dailymail': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/cnn_dailymail.jsonl.gz',
'agnews': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/agnews.jsonl.gz',
'yahoo_answers_title_answer': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/yahoo_answers_title_answer.jsonl.gz',
'S2ORC_title_abstract': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/S2ORC_title_abstract.jsonl.gz',
'flickr30k_captions': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/flickr30k_captions.jsonl.gz',
'sentence-compression': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/sentence-compression.jsonl.gz',
'PAQ_pairs': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/PAQ_pairs.jsonl.gz',
'xsum': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/xsum.jsonl.gz',
'amazon_review_2018': 'https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/amazon_review_2018.jsonl.gz'
}

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True,truncation=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
    return input_tokens

def download_data(data_dir,url):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir,exist_ok=True)
    os.system(f"wget {url}")
    file_name = url.split('/')[-1]
    os.system(f"gzip -d {file_name}")
    file_name = file_name[:-len('.gz')]
    shutil.move(file_name, data_dir)

def deduct_k_largest(tensor, k):
    values, indices = torch.topk(tensor, k, dim=1)
    result = torch.zeros_like(tensor)
    values += 1000
    for i in range(tensor.size(0)):
        result[i, indices[i]] = values[i]
    return result

def compute_similarity(q_reps, p_reps):
    if len(p_reps.size()) == 2: return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))

def pooling(token_embeddings, attention_mask,weighted_mean=True):
    if weighted_mean:
        attention_mask *= attention_mask.cumsum(dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_llama2_embedding(model,tokenizer,texts,batch_size):
    print('calculate llama2 embedding')
    representation = []
    with torch.no_grad():
        for start_idx in trange(0,len(texts),batch_size):
            cur_texts = texts[start_idx:start_idx+batch_size]
            input_tokens = prepare_input(tokenizer=tokenizer, prompts=cur_texts)
            psg_out = model(**input_tokens, return_dict=True)
            p_reps = pooling(psg_out.last_hidden_state, input_tokens['attention_mask']).cpu()
            del psg_out
            representation.append(p_reps)
    representation = torch.cat(representation, 0).cpu()
    return torch.nn.functional.normalize(representation, dim=-1)

def calculate_bge_embedding(model,texts,batch_size,**kwargs):
    emb = model.encode(texts, normalize_embeddings=True,show_progress_bar=True,batch_size=batch_size)
    emb = torch.nn.functional.normalize(torch.from_numpy(emb), dim=-1)
    return emb

def calculate_instructor_embedding(model,texts,batch_size,**kwargs):
    emb = model.encode(texts, show_progress_bar=True,batch_size=batch_size)
    emb = torch.nn.functional.normalize(torch.from_numpy(emb), dim=-1)
    return emb

def get_emb(cache_file, model, tokenizer, texts, batch_size,calculate_function):
    if cache_file is not None and os.path.isfile(cache_file):
        emb = torch.load(cache_file)
    else:
        emb = calculate_function(model=model,tokenizer=tokenizer,texts=texts,batch_size=batch_size)
        if cache_file is not None:
            cur_cache_dir = '/'.join(cache_file.split('/')[:-1])
            if not os.path.isdir(cur_cache_dir):
                os.makedirs(cur_cache_dir, exist_ok=True)
            torch.save(emb, cache_file)
    return emb

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True,)
    parser.add_argument("--student_model", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--student_model_max_length", type=int, default=2048)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_num", type=int, default=-1)
    parser.add_argument("--teacher_model_batch_size", type=int, default=32)
    parser.add_argument("--student_model_batch_size", type=int, default=4)
    parser.add_argument("--process_batch_size", type=int, default=10000)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--hard_negative_mine", type=str, default=None)
    # config should have keys:
    # instruction_for_instructor_encode_query (string), instruction_for_instructor_encode_doc (string),
    # query_instructions (list of string), doc_instructions (list of string)
    parser.add_argument("--data_id", type=str, required=True,
                        choices=['multi-train/emb-fever-train', 'multi-train/emb-gigaword',
                                 'multi-train/emb-hotpotqa-train',
                                 'multi-train/emb-medmcqa-train', 'multi-train/emb-multi-lexsum',
                                 'multi-train/emb-nq-train',
                                 'multi-train/emb-pubmed', 'multi-train/emb-record_train',
                                 'multi-train/emb-reddit-title-body',
                                 'multi-train/emb-scitldr', 'multi-train/emb-trex-train',
                                 'multi-train/emb-triviaqa-train',
                                 'multi-train/emb-wow-train', 'multi-train/emb-zeroshot-train',
                                 'embedding-data/QQP_triplets','flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl',
                                 'specter_train_triples','AllNLI','msmarco-triplets','codesearchnet','WikiAnswers',
                                 'altlex','SimpleWiki','amazon-qa','eli5_question_answer','gooaq_pairs','squad_pairs',
                                 'searchQA_top5_snippets','coco_captions','ccnews_title_text','npr','S2ORC_title_abstract',
                                 'wikihow','cnn_dailymail','agnews','yahoo_answers_title_answer','flickr30k_captions',
                                 'sentence-compression','PAQ_pairs','xsum','amazon_review_2018'
                                 ])
    args = parser.parse_args()
    task_name = args.data_id.split('-')[-1]

    with open(args.config_file) as f:
        config = json.load(f)

    data_tripples = []
    if args.data_id=='embedding-data/QQP_triplets':
        raw_data = load_dataset('embedding-data/QQP_triplets')['train']
        for e in raw_data:
            if args.max_num > 0 and len(data_tripples)>=args.max_num:
                break
            assert isinstance(e["query"],str) and isinstance(e["pos"],list) and isinstance(e["pos"][0],str) and isinstance(e["neg"],list) and isinstance(e["neg"][0],str)
            data_tripples.append([e["query"],e["pos"][0],e['neg'][0]])
    elif args.data_id=='flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl':
        subsets = ['english', 'academia', 'christianity', 'apple', 'electronics', 'gaming', 'askubuntu', 'ell', 'hermeneutics',
         'judaism', 'diy', 'law', 'history', 'islam', 'dba', 'cooking', 'gamedev', 'drupal', 'chemistry', 'android',
         'mathoverflow', 'magento', 'buddhism', 'gis', 'graphicdesign', 'codereview', 'aviation', 'bicycles',
         'japanese', 'cs', 'german', 'interpersonal', 'biology', 'bitcoin', 'blender', 'crypto', 'anime', 'boardgames',
         'hinduism', 'french', 'fitness', 'economics', 'chinese', 'codegolf', 'linguistics', 'astronomy', 'arduino',
         'chess', 'cstheory', 'ja', 'martialarts', 'mathematica', 'dsp', 'ethereum', 'health', 'cogsci', 'earthscience',
         'gardening', 'datascience', 'literature', 'matheducators', 'lifehacks', 'engineering', 'ham', '3dprinting',
         'italian', 'emacs', 'homebrew', 'ai', 'avp', 'expatriates', 'elementaryos', 'cseducators', 'hsm',
         'expressionengine', 'joomla', 'freelancing', 'crafts', 'genealogy', 'latin', 'hardwarerecs', 'devops',
         'coffee', 'beer', 'languagelearning', 'ebooks', 'bricks', 'civicrm', 'bioinformatics', 'esperanto',
         'computergraphics', 'conlang', 'korean', 'iota', 'eosio', 'craftcms', 'iot', 'drones', 'cardano', 'materials',
         'ru', 'softwareengineering', 'scifi', 'workplace', 'serverfault', 'rpg', 'physics', 'superuser',
         'worldbuilding', 'security', 'pt', 'unix', 'meta', 'politics', 'stats', 'movies', 'photo', 'wordpress',
         'music', 'philosophy', 'skeptics', 'money', 'salesforce', 'parenting', 'raspberrypi', 'travel', 'mechanics',
         'tex', 'ux', 'sharepoint', 'webapps', 'puzzling', 'networkengineering', 'webmasters', 'sports', 'rus', 'space',
         'writers', 'pets', 'pm', 'russian', 'spanish', 'sound', 'quant', 'sqa', 'outdoors', 'softwarerecs',
         'retrocomputing', 'mythology', 'portuguese', 'opensource', 'scicomp', 'ukrainian', 'patents', 'sustainability',
         'poker', 'robotics', 'woodworking', 'reverseengineering', 'sitecore', 'tor', 'vi', 'windowsphone',
         'vegetarianism', 'moderators', 'quantumcomputing', 'musicfans', 'tridion', 'opendata', 'tezos', 'stellar',
         'or', 'monero', 'stackapps']
        for split in subsets:
            raw_data = load_dataset('flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl',split)['train']
            for e in raw_data:
                if args.max_num > 0 and len(data_tripples) >= args.max_num:
                    break
                assert isinstance(e["title_body"], str) and isinstance(e["upvoted_answer"],str) and isinstance(e["downvoted_answer"], str)
                data_tripples.append([e["title_body"], e["upvoted_answer"], e["downvoted_answer"]])
    elif args.data_id in ['specter_train_triples','AllNLI']:
        if not os.path.isfile(os.path.join(args.data_dir,f'{args.data_id}.jsonl')):
            download_data(data_dir=args.data_dir,url=URLS[args.data_id])
        with open(os.path.join(args.data_dir,f'{args.data_id}.jsonl')) as f:
            for line in f:
                example = json.loads(line)
                assert isinstance(example,list) and len(example)==3 and isinstance(example[0],str) and isinstance(example[1],str) and isinstance(example[2],str)
                data_tripples.append(example)
    elif args.data_id in ['msmarco-triplets']:
        if not os.path.isfile(os.path.join(args.data_dir,f'{args.data_id}.jsonl')):
            download_data(data_dir=args.data_dir,url=URLS[args.data_id])
        with open(os.path.join(args.data_dir,f'{args.data_id}.jsonl')) as f:
            for line in f:
                e = json.loads(line)
                if args.max_num > 0 and len(data_tripples) >= args.max_num:
                    break
                assert isinstance(e["query"], str) and isinstance(e["pos"], list) and isinstance(e["pos"][0],str) \
                       and isinstance(e["neg"], list) and isinstance(e["neg"][0], str)
                data_tripples.append([e["query"], e["pos"][0], e['neg'][0]])
    elif args.data_id in ['multi-train/emb-fever-train','multi-train/emb-hotpotqa-train','multi-train/emb-nq-train',
                          'multi-train/emb-trex-train','multi-train/emb-triviaqa-train','multi-train/emb-wow-train',
                          'multi-train/emb-zeroshot-train']:
        # data triples
        raw_data = load_dataset(args.data_id, token='YOUR_HF_KEY')['train']
        for e in raw_data:
            data_tripples.append([e['query'],e['pos'],e['neg']])
    elif args.data_id in ['multi-train/emb-gigaword','multi-train/emb-medmcqa-train','multi-train/emb-multi-lexsum',
                          'multi-train/emb-pubmed','multi-train/emb-record_train','multi-train/emb-reddit-title-body',
                          'multi-train/emb-scitldr','codesearchnet','WikiAnswers','altlex','SimpleWiki','amazon-qa',
                          'eli5_question_answer','gooaq_pairs','squad_pairs','coco_captions','ccnews_title_text','npr',
                          'S2ORC_title_abstract','wikihow','cnn_dailymail','agnews','yahoo_answers_title_answer',
                          'flickr30k_captions','sentence-compression','PAQ_pairs','xsum','amazon_review_2018']:
        # data pairs
        if args.data_id in ['multi-train/emb-gigaword', 'multi-train/emb-medmcqa-train', 'multi-train/emb-multi-lexsum',
                            'multi-train/emb-pubmed', 'multi-train/emb-record_train',
                            'multi-train/emb-reddit-title-body', 'multi-train/emb-scitldr']:
            raw_data = load_dataset(args.data_id, token='YOUR_HF_KEY')['train']
            features = set(raw_data.features.keys())
            assert features == {'query', 'pos', 'idx', 'task_name'}
            queries = [e['query'] for e in raw_data]
            docs = [e['pos'] for e in raw_data]
            assert len(queries) == len(docs)
            if args.max_num > 0:
                queries = queries[:args.max_num]
                docs = docs[:args.max_num]
        elif args.data_id in ['codesearchnet','eli5_question_answer','gooaq_pairs','squad_pairs','ccnews_title_text',
                              'npr','S2ORC_title_abstract','wikihow','cnn_dailymail','agnews','amazon_review_2018',
                              'yahoo_answers_title_answer','sentence-compression','PAQ_pairs','xsum']:
            if not os.path.isfile(os.path.join(args.data_dir, f'{args.data_id}.jsonl')):
                download_data(data_dir=args.data_dir, url=URLS[args.data_id])
            queries = []
            docs = []
            with open(os.path.join(args.data_dir, f'{args.data_id}.jsonl')) as f:
                for line in f:
                    e = json.loads(line)
                    assert isinstance(e,list) and isinstance(e[0],str) and isinstance(e[1],str)
                    queries.append(e[0])
                    docs.append(e[1])
                    if args.max_num > 0 and len(queries)>=args.max_num:
                        break
        elif args.data_id in ['WikiAnswers','altlex','SimpleWiki','coco_captions','flickr30k_captions']:
            if not os.path.isfile(os.path.join(args.data_dir, f'{args.data_id}.jsonl')):
                download_data(data_dir=args.data_dir, url=URLS[args.data_id])
            queries = []
            docs = []
            with open(os.path.join(args.data_dir, f'{args.data_id}.jsonl')) as f:
                for line in f:
                    e = json.loads(line)
                    assert isinstance(e['set'], list) and isinstance(e['set'][0], str) and isinstance(e['set'][1], str)
                    queries.append(e['set'][0])
                    docs.append(e['set'][1])
                    if args.max_num > 0 and len(queries) >= args.max_num:
                        break
        elif args.data_id in ['amazon-qa','searchQA_top5_snippets']:
            if not os.path.isfile(os.path.join(args.data_dir, f'{args.data_id}.jsonl')):
                download_data(data_dir=args.data_dir, url=URLS[args.data_id])
            queries = []
            docs = []
            with open(os.path.join(args.data_dir, f'{args.data_id}.jsonl')) as f:
                for line in f:
                    e = json.loads(line)
                    assert isinstance(e['query'], str) and isinstance(e['pos'], list) and isinstance(e['pos'][0], str)
                    queries.append(e['query'])
                    docs.append(e['pos'][0])
                    if args.max_num > 0 and len(queries) >= args.max_num:
                        break
        else:
            raise ValueError(f"{args.data_id} is not supported yet")

        negative_map = {}
        total_num = len(queries)
        if args.cache_dir and os.path.isfile(os.path.join(args.cache_dir,'negative_maps',task_name,'negative_map.jsonl')):
            with open(os.path.join(args.cache_dir,'negative_maps',task_name,'negative_map.jsonl')) as f:
                for line in f:
                    cur_example = json.loads(line)
                negative_map[cur_example[0]] = cur_example[1]
        else:
            student_model = AutoModel.from_pretrained(args.student_model, torch_dtype=torch.float16,
                                                     token='YOUR_HF_KEY').cuda()
            tokenizer = AutoTokenizer.from_pretrained(args.student_model, use_fast=False,
                                                             model_max_length=args.student_model_max_length, padding_side="right")
            tokenizer.pad_token = tokenizer.unk_token
            student_model_id = args.student_model.split('/')[-1]
            teacher_model_id = args.teacher_model.split('/')[-1]

            doc_emb_student = get_emb(
                cache_file=os.path.join(args.cache_dir,'emb_cache',task_name,f'doc_emb_{student_model_id}.pt') if args.cache_dir is not None else None,
                model=student_model,
                tokenizer=tokenizer,
                texts=docs,
                batch_size=args.student_model_batch_size,
                calculate_function=calculate_llama2_embedding
            )
            if 'bge' in args.teacher_model:
                teacher_model = SentenceTransformer(args.teacher_model)
                doc_emb_teacher = get_emb(
                    cache_file=os.path.join(args.cache_dir, 'emb_cache', task_name, f'doc_emb_{teacher_model_id}.pt') if args.cache_dir is not None else None,
                    model=teacher_model,
                    tokenizer=tokenizer,
                    texts=docs,
                    batch_size=args.teacher_model_batch_size,
                    calculate_function=calculate_instructor_embedding
                )
            elif 'instructor' in args.teacher_model:
                # queries_inst = [[config['instruction_for_instructor_encode_query'], q] for q in queries]
                docs_inst = [[config['instruction_for_instructor_encode_doc'], d] for d in docs]
                teacher_model = INSTRUCTOR(args.teacher_model)
                doc_emb_teacher = get_emb(
                    cache_file=os.path.join(args.cache_dir, 'emb_cache', task_name, f'doc_emb_{teacher_model_id}.pt') if args.cache_dir is not None else None,
                    model=teacher_model,
                    tokenizer=tokenizer,
                    texts=docs_inst,
                    batch_size=args.teacher_model_batch_size,
                    calculate_function=calculate_instructor_embedding
                )
            else:
                raise ValueError(f"{args.teacher_model} is not supported yet")
            negative_strategy,cur_value = args.hard_negative_mine.split('-')
            if negative_strategy in ['percent','similarity_value']:
                cur_value = float(cur_value)
                if negative_strategy=='percent':
                    assert negative_strategy>0 and negative_strategy<1
            elif negative_strategy in ['topk']:
                cur_value = int(cur_value)
            else:
                raise ValueError(f"Hard negative mining strategy {negative_strategy} is not supported yet")
            for start_idx in range(0, total_num, args.process_batch_size):
                if 'bge' in args.student_model:
                    query_emb_teacher = calculate_bge_embedding(
                        model=teacher_model,
                        texts=queries[start_idx:start_idx + args.process_batch_size],
                        batch_size=args.teacher_model_batch_size
                    )
                elif 'instructor' in args.teacher_model:
                    query_emb_teacher = calculate_instructor_embedding(
                        model=teacher_model,
                        texts=[[config['instruction_for_instructor_encode_query'], q] for q in queries[start_idx:start_idx + args.process_batch_size]],
                        batch_size=args.teacher_model_batch_size
                    )
                else:
                    raise ValueError(f"{args.teacher_model} is not supported yet")
                query_emb_student = calculate_llama2_embedding(model=student_model, tokenizer=tokenizer,
                                                              texts=queries[start_idx:start_idx + args.process_batch_size],
                                                              batch_size=args.student_model_batch_size)
                teacher_scores = compute_similarity(query_emb_teacher, doc_emb_teacher)
                student_scores = compute_similarity(query_emb_student, doc_emb_student)
                if negative_strategy=='similarity_value':
                    mask_matrix = (teacher_scores > cur_value).long() * 1000
                elif negative_strategy=='percent':
                    threshold = int(len(student_scores[0])*cur_value)
                    mask_matrix = deduct_k_largest(teacher_scores,threshold)
                elif negative_strategy=='topk':
                    mask_matrix = deduct_k_largest(teacher_scores, cur_value)
                else:
                    raise ValueError(f"Hard negative mining strategy {negative_strategy} is not supported yet")
                scores = student_scores-teacher_scores
                cur_queries = queries[start_idx:start_idx + args.process_batch_size]
                assert len(cur_queries) == len(scores), f"{len(cur_queries)}, {len(scores)}"
                assert len(docs) == len(scores[0]), f"{len(docs)}, {len(scores[0])}"
                score_idx = 0
                for q, s in zip(cur_queries, scores):
                    assert queries[start_idx + score_idx] == q
                    cur_doc = docs[start_idx + score_idx]
                    selected_idx = np.argmax(s)
                    while docs[selected_idx]==cur_doc:
                        s[selected_idx] = -10000
                        selected_idx = np.argmax(s)
                    score_idx += 1
                    negative_map[q] = docs[selected_idx]
                    assert cur_doc != negative_map[q]
            if args.cache_dir is not None:
                if not os.path.isdir(os.path.join(args.cache_dir,'negative_maps',task_name)):
                    os.makedirs(os.path.join(args.cache_dir,'negative_maps',task_name),exist_ok=True)
                with open(os.path.join(args.cache_dir,'negative_maps',task_name,'negative_map.jsonl'),'w') as f:
                    for query1, negative1 in negative_map.items():
                        assert isinstance(query1, str) and isinstance(negative1, str)
                        f.write(json.dumps([query1, negative1]) + '\n')
        assert len(queries)==len(docs)
        for q,d in zip(queries,docs):
            data_tripples.append([q,d,negative_map[q]])
    else:
        raise ValueError(f"{args.data_id} is not supported yet")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,f"{task_name}.jsonl"),'w') as f:
        for example in data_tripples:
            assert isinstance(example,list) and len(example)==3 and isinstance(example[0],str) and isinstance(example[1],str) and isinstance(example[2],str)
            cur_dict = {
                'query': example[0],
                'pos': [example[1]],
                'neg': [example[2]],
                'task': task_name,
                'instructions': {
                    'query': random.choice(config['query_instructions']),
                    'pos': random.choice(config['doc_instructions']),
                    'neg':  random.choice(config['doc_instructions']),
                }
            }
            f.write(json.dumps(cur_dict)+'\n')
