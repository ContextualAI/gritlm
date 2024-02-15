import argparse
from collections import defaultdict
import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from rag import dist_utils
from rag.index import load_or_initialize_index
from rag.tasks import get_task
from gritlm import GritLM


EMBED_BOS = "<|embed|>\n"

NO_RETRIEVAL = "<|user|>\n{query}\n<|assistant|>\n"

FULL_FORMAT = "<|embed|>\n{query}\n<|user|>\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
FULL_FORMAT_NO_EMBED = "<|user|>\n{query}\n\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"

FULL_FORMAT_DOC = "<|embed|>\n{title} {text}\n<|user|>\n{query}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
FULL_FORMAT_NO_EMBED_DOC = "<|user|>\n{title} {text}\n\n{query}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"

CACHE_FORMAT_QUERY = "\n<|user|>\n{title} {text}\n\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"
CACHE_FORMAT_DOC = "\n<|user|>\n{query}\n\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
CACHE_FORMAT_DOC_QUERY = "\n<|user|>\nAnswer the prior query while optionally using the context prior to it\n<|assistant|>\n"
CACHE_FORMAT_QUERY_DOC = "\n<|user|>\nOptionally using the prior context answer the query prior to it\n<|assistant|>\n"

PROMPT = "The answer is"
#PROMPT = "Sure, the answer is"

def gritlm_instruction_format(instruction=None):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="gritlm", type=str)
    parser.add_argument('--task_names', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument(
        "--load_index_path",
        default=None,
        type=str,
        help="path for loading the index, passage embeddings and passages",
    )
    parser.add_argument(
        "--save_index_path",
        default=None,
        type=str,
        help="path for saving the index and/or embeddings",
    )
    parser.add_argument(
        "--passages",
        nargs="+",
        help="list of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path",
    )
    parser.add_argument(
        "--save_index_n_shards",
        default=1,
        type=int,
        help="how many shards to save an index to file with. Must be an integer multiple of the number of workers.",
    )
    parser.add_argument(
        "--eval_data",
        nargs="+",
        default=[],
        help="list of space-separated paths to jsonl-formatted evaluation sets",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="qa",
        choices=["qa"],
    )    
    parser.add_argument("--n_context", type=int, default=1, help="number of top k passages to pass to reader")    
    parser.add_argument("--limit", type=int, default=None, help="limit number of passages to index")
    parser.add_argument("--limit_start", type=int, default=0, help="denote start of limit")
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)        
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="None / query / doc / querydoc / docquery",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="default",
        help="",
    )    
    parser.add_argument(
        "--per_gpu_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU.",
    )
    parser.add_argument(
        "--embedbs",
        default=128,
        type=int,
        help="Batch size for embedding docs.",
    )
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--customd', default=None, type=str)
    parser.add_argument('--customq', default=None, type=str)
    parser.add_argument('--cut_embed_bos', action='store_true', help="Cut the embed bos token from the cache")
    parser.add_argument('--idxdtype', default="float32", type=str, help="Index dtype")
    parser.add_argument('--pooling_method', default="mean", type=str, help="GritLM Attn mode")
    parser.add_argument('--attn', default="bbcc", type=str, help="GritLM Attn mode")
    parser.add_argument('--attn_implementation', default="sdpa", type=str, help="GritLM Attn imp")
    parser.add_argument('--no_retrieval', action='store_true', help="No retrieval")
    parser.add_argument('--move_cache_to_cpu', action='store_true', help="Move doc cache to cpu")
    parser.add_argument('--latency', action='store_true', help="Move doc cache to cpu")
    return parser.parse_args()

@torch.no_grad()
def build_index(model, index, passages, gpu_embedder_batch_size=512, cache=False, move_cache_to_cpu=False):
    n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
    total = 0
    for i in range(n_batch):
        batch = passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
        if cache:
            # kv_cache: Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`
            embeddings, kv_cache = model.encode_corpus(
                batch,
                get_cache=True,
                convert_to_tensor=True,
                instruction=gritlm_instruction_format(),
                add_special_tokens=False,
            )
            if move_cache_to_cpu:
                kv_cache = [(layer[0].cpu(), layer[1].cpu()) for layer in kv_cache]
            for j in range(0, kv_cache[0][0].shape[0]):
                index.doc_map[total+j]["kv_cache"] = [[k_or_v[j:j+1] for k_or_v in layer] for layer in kv_cache]
        else:
            embeddings = model.encode_corpus(batch, convert_to_tensor=True, instruction=gritlm_instruction_format())
        index.embeddings[:, total : total + len(embeddings)] = embeddings.T.to(index.dtype)
        total += len(embeddings)
        if i % 500 == 0 and i > 0:
            logger.info(f"Number of passages encoded: {total}")
    dist_utils.barrier()
    logger.info(f"{total} passages encoded on process: {dist_utils.get_rank()}")

def _get_eval_data_iterator(args, data_path, task):
    data_iterator = task.data_iterator(data_path)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, args.per_gpu_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator

@torch.no_grad()
def evaluate(model, index, opt, data_path):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []

    task = get_task(opt, model.tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)
    if opt.limit:
        data_iterator = data_iterator[:opt.limit]
    if opt.customq:
        # data_iterator = [max(data_iterator, key=lambda x: len(x["query"][0]))]
        if os.path.exists(opt.customq):
            with open(opt.customq, "r") as f:
                data_iterator[0]["query"] = [f.read()]            
        else:
            # Is number
            data_iterator[0]["query"] = ["<s>" * int(opt.customq)]
        # Expand to 100 times for standard dev
        for i in range(99):
            data_iterator.append(data_iterator[0])

    times = []
    time_to_remove = 0
    for i, batch in enumerate(tqdm(data_iterator)):
        query = batch.get("query", [""])
        answers = batch.get("answers", [""])
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0): continue

        start_time = time.time()
        if opt.no_retrieval is False:
            if (args.cache is not None) and ("query" in args.cache):
                query_emb, kv_cache = model.encode_queries(
                    query,
                    get_cache=True,
                    convert_to_tensor=True,
                    #instruction="\n" + gritlm_instruction_format() if args.cache == "docquery" else gritlm_instruction_format(),
                    #add_special_tokens=False if args.cache == "docquery" else True,
                    # Even if docquery, it performs better in the regular format
                    instruction=gritlm_instruction_format(),
                    add_special_tokens=True,
                )
            else:
                query_emb = model.encode_queries(
                    query,
                    convert_to_tensor=True,
                    instruction=gritlm_instruction_format(),
                    add_special_tokens=True,
                )
                kv_cache = None
            passages, scores = index.search_knn(query_emb, args.n_context)
            assert (len(passages) == 1) and (len(passages[0]) == 1), "Only 1 passage per query supported for now"

            # Add doc cache on the fly (this makes no sense for production but is useful for benchmarking)
            if (args.cache is not None) and ("doc" in args.cache) and ("kv_cache" not in passages[0][0]):
                # In production this is cached so should not be timed
                time_to_remove = time.time()
                passages[0][0]["kv_cache"] = model.encode_corpus(
                    [passages[0][0]],
                    get_cache=True,
                    convert_to_tensor=False,
                    # Add newline to ensure formatting remains intact
                    instruction="\n" + gritlm_instruction_format() if args.cache == "querydoc" else gritlm_instruction_format(),
                    add_special_tokens=False if args.cache == "querydoc" else True,
                )[1]
                time_to_remove = time.time() - time_to_remove

            if args.cache == "query":
                inputs = CACHE_FORMAT_QUERY.format(**passages[0][0])
            elif args.cache == "doc":
                inputs = CACHE_FORMAT_DOC.format(query=query[0])
                kv_cache = [(
                    passages[0][0]["kv_cache"][i][0].to(query_emb.device),
                    passages[0][0]["kv_cache"][i][1].to(query_emb.device),
                ) for i in range(len(passages[0][0]["kv_cache"]))]
            elif args.cache == "docquery":
                inputs = CACHE_FORMAT_DOC_QUERY
                # Concat the doc kv cache prior to the query kv cache
                # Inaccuracy is that the query kv cache is not conditioned on the doc kv cache
                kv_cache = [(
                    torch.cat((passages[0][0]["kv_cache"][i][0], layer[0]), dim=2).to(query_emb.device),
                    torch.cat((passages[0][0]["kv_cache"][i][1], layer[1]), dim=2).to(query_emb.device),
                ) for i, layer in enumerate(kv_cache)]
            elif args.cache == "querydoc":
                inputs = CACHE_FORMAT_QUERY_DOC
                # Concat the query cache prior to the doc kv cache
                # Inaccuracy is that the doc kv cache is not conditioned on the query kv cache
                kv_cache = [(
                    torch.cat((layer[0], passages[0][0]["kv_cache"][i][0]), dim=2).to(query_emb.device),
                    torch.cat((layer[1], passages[0][0]["kv_cache"][i][1]), dim=2).to(query_emb.device),
                ) for i, layer in enumerate(kv_cache)]
            elif args.cache is None:
                if args.prompt == "queryemb":
                    inputs = FULL_FORMAT.format(query=query[0], **passages[0][0])
                elif args.prompt == "docemb":
                    inputs = FULL_FORMAT_DOC.format(query=query[0], **passages[0][0])
                elif args.prompt in ("query", "default"):
                    inputs = FULL_FORMAT_NO_EMBED.format(query=query[0], **passages[0][0])                
                elif args.prompt == "doc":
                    inputs = FULL_FORMAT_NO_EMBED_DOC.format(query=query[0], **passages[0][0])
            else:
                raise ValueError(f"Cache type {args.cache} not supported")
        else:
            kv_cache = None
            inputs = NO_RETRIEVAL.format(query=query[0])

        inputs += PROMPT

        inputs = model.tokenizer(
            inputs,
            padding=False if args.latency else True,
            truncation=False if args.latency else True,
            return_tensors="pt",
            max_length=None if args.latency else 4096,
            # bos token is already in the cache
            add_special_tokens=False if (args.cache is not None or args.latency) else True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        if args.cache is not None:
            inputs["use_cache"] = True
            # Attend to the cache too
            inputs["attention_mask"] = torch.cat((
                torch.ones((kv_cache[0][0].shape[0], kv_cache[0][0].shape[2]), dtype=torch.long, device=inputs["attention_mask"].device),
                inputs["attention_mask"],
            ), dim=1)
        generation = model.generate(
            **inputs,
            min_new_tokens=opt.min_new_tokens,
            max_new_tokens=opt.max_new_tokens,
            past_key_values=kv_cache,
            pad_token_id=model.tokenizer.pad_token_id,
        )
        times.append(time.time() - start_time - time_to_remove)

        generation = generation[0][len(inputs["input_ids"][0]):]
        pred = model.tokenizer.decode(generation, skip_special_tokens=True)
        # Use em as main key since if em=1, then match & f1 are also 1
        sample_metrics = max(
            [task.evaluation(pred, gold) for gold in answers], key=lambda x: (x["exact_match"], x["match"], x["f1"])
        )

        for key, value in sample_metrics.items():
            metrics[key].append(value)

        ex = {"query": query[0], "answers": answers, "generation": pred}
        # Add per-sample metrics
        ex = {**ex, **sample_metrics}
        if opt.no_retrieval is False:
            if (args.cache is not None) and ("doc" in args.cache):
                passage_keys = set(passages[0][0].keys()) - set(["kv_cache"])
                ex["passages"] = [{k: passages[0][i][k] for k in passage_keys} for i in range(len(passages[0]))]
            else:
                ex["passages"] = passages[0]
        if batch_metadata is not None:
            ex["metadata"] = batch_metadata[0]
        if "id" in batch:
            ex["id"] = batch["id"][0]
        dataset_wpred.append(ex)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = dist_utils.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}

    dataset_name, _ = os.path.splitext(os.path.basename(data_path))
    dataset_name = f"{dataset_name}-{args.cache if args.cache is not None else 'nocache'}-{args.max_new_tokens}maxtoks-{args.prompt}prompt"
    if args.no_retrieval:
        dataset_name += "-noretrieval"
    model_name = args.model_name_or_path.split("/")[-1]
    save_dir = opt.save_dir if opt.save_dir is not None else f"gritlmresults/{model_name}"

    avg_time_ex = sum(times) / len(times)
    std_time_ex = float(torch.tensor(times).std())
    total_time = sum(times)

    if args.latency:
        latency_path = f"{save_dir}/{dataset_name}-latency.json"
        if os.path.exists(latency_path):
            logger.info(f"Loading latency results from {latency_path}")
            with open(latency_path, "r") as f:
                latency = json.load(f)
        else:
            latency = {}
        addon = "gpu" if torch.cuda.is_available() else "cpu"
        latency[str(opt.customq) + "-" + str(opt.customd) + "-" + str(args.max_new_tokens) + "-" + addon] = {
            "avg": avg_time_ex, 
            "std": std_time_ex, 
            "total": total_time,
            "q_len": opt.customq,
            "d_len": opt.customd,
            "max_toks": args.max_new_tokens,
            "device": addon,
            "inputs_shape": tuple(inputs["input_ids"].shape),
        }
        with open(latency_path, "w") as f:
            json.dump(latency, f, indent=4)
        logger.info(f"Latency results saved to {latency_path}")
    else:
        dist_utils.save_distributed_dataset(dataset_wpred, dataset_name, dist_utils.get_rank(), save_dir)

    logger.info(f"Average time per example (s): {avg_time_ex}")
    logger.info(f"Standard deviation of time per example (s): {std_time_ex}")
    logger.info(f"Total time (s): {total_time}")
    return metrics

if __name__ == "__main__":
    args = get_args()
    if (args.cache is not None) and (args.cache == "None"):
        args.cache = None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.no_retrieval is False:
        index, passages = load_or_initialize_index(args, logger, dim=4096)
    else:
        index, passages = None, None
    pool = "mean" if "bb" in args.model_name_or_path else "weightedmean"

    gritlm_kwargs = {
        "pooling_method": args.pooling_method,
        "attn": args.attn,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": torch.bfloat16,
    }

    if (args.load_index_path is None) and (args.no_retrieval is False):
        gritlm_kwargs["mode"] = "embedding"

    if args.max_length is not None:
        gritlm_kwargs["max_length"] = args.max_length

    model = GritLM(args.model_name_or_path, **gritlm_kwargs)
    model.eval()

    if not(model.tokenizer.pad_token) and model.tokenizer.bos_token:
        model.tokenizer.pad_token = model.tokenizer.bos_token
        logger.info('Set pad token to bos token: %s', tokenizer.pad_token)
    
    if args.cut_embed_bos:
        embed_bos_tokens = model.tokenizer.encode(EMBED_BOS, add_special_tokens=False)

    if (args.load_index_path is None) and (args.no_retrieval is False):
        build_index(
            model,
            index,
            passages,
            gpu_embedder_batch_size=args.embedbs,
            cache=args.cache is not None and ("doc" in args.cache),
            move_cache_to_cpu=args.move_cache_to_cpu,
        )

    if args.save_index_path is not None:
        os.makedirs(args.save_index_path, exist_ok=True)
        index.save_index(args.save_index_path, args.save_index_n_shards)


    if (args.load_index_path is None) and (args.no_retrieval is False):
        # Reload without DP
        del model
        gritlm_kwargs["mode"] = "unified"
        model = GritLM(args.model_name_or_path, **gritlm_kwargs)

    for data_path in args.eval_data:
        dataset_name = os.path.basename(data_path)

        # Skip if exists
        if args.latency:
            dataset_name, _ = os.path.splitext(os.path.basename(data_path))
            dataset_name = f"{dataset_name}-{args.cache if args.cache is not None else 'nocache'}-{args.max_new_tokens}maxtoks-{args.prompt}prompt"
            if args.no_retrieval:
                dataset_name += "-noretrieval"
            model_name = args.model_name_or_path.split("/")[-1]
            save_dir = args.save_dir if args.save_dir is not None else f"gritlmresults/{model_name}"
            latency_path = f"{save_dir}/{dataset_name}-latency.json"
            if os.path.exists(latency_path):
                logger.info(f"Loading from {latency_path}.")
                with open(latency_path, "r") as f:
                    latency = json.load(f)
            else:
                latency = {}
            addon = "gpu" if torch.cuda.is_available() else "cpu"
            if str(args.customq) + "-" + str(args.customd) + "-" + str(args.max_new_tokens) + "-" + addon in latency:
                logger.info(f"Latency results for {args.customq}-{args.customd}-{args.max_new_tokens}-{addon} already exist")
                continue

        logger.info(f"Start Evaluation on {data_path}")
        print(f"MIN / MAX TOKS: {args.min_new_tokens} / {args.max_new_tokens}")
        metrics = evaluate(model, index, args, data_path)
        log_message = f"Dataset: {dataset_name}"
        for k, v in metrics.items():
            log_message += f" | {v:.3f} {k}"
        logger.info(log_message)

