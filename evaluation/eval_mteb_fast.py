import argparse
import os
from functools import partial

from mteb import MTEB
from sentence_transformers import SentenceTransformer
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="GritLM/GritLM-7B", type=str)
    parser.add_argument('--attn_implementation', default='sdpa', type=str, help="eager/sdpa/flash_attention_2")
    parser.add_argument('--attn', default='bbcc', type=str, help="only first two letters matter for embedding")
    parser.add_argument('--task_types', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--task_names', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--instruction_set', default="e5", type=str, help="Instructions to use")
    parser.add_argument('--instruction_format', default="gritlm", type=str, help="Formatting to use")
    parser.add_argument('--no_instruction', action='store_true', help="Do not use instructions")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--num_shots', default=None, type=int)
    parser.add_argument('--dtype', default='bfloat16', type=str)
    parser.add_argument('--output_folder', default="results", type=str)
    parser.add_argument('--overwrite_results', action='store_true')
    parser.add_argument('--pipeline_parallel', action='store_true')
    parser.add_argument('--embedding_head', default=None, type=str)
    parser.add_argument('--pooling_method', default='mean', type=str)
    parser.add_argument('--save_qrels', action='store_true')
    parser.add_argument('--second_to_last_hidden', default='False', action='store_true')    
    parser.add_argument('--top_k', default=10, type=int)    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    task_name = args.task_names
    output_folder = args.output_folder


    model = SentenceTransformer(args.model_name_or_path)
    eval_splits = ["test" if task_name not in ['MSMARCO', 'Ko-miracl'] else 'dev']
    evaluation = MTEB(tasks=[task_name], task_langs=['en'])
    evaluation.run(
        model,
        output_folder=output_folder,
        eval_splits=eval_splits,
        batch_size=args.batch_size,
        save_qrels=args.save_qrels,
        overwrite_results=args.overwrite_results,
    )
