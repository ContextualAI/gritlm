import copy
import json
import logging
from time import time
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings
import os

from .AbsTask import AbsTask
from tqdm import tqdm

logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]

TEMPLATES = {
     "ArguAna": "<|user|>\n" \
                "Provided two debate paragraphs, check if they are about the same topic, but contain counter-arguments.\n\n" \
                "Paragraph 1: {query}\n" \
                "Paragraph 2: {passage}\n\n" \
                "Answer with yes if paragraph 1 and paragraph 2 are about the same topic, but contain counter-arguments; Answer with no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "SciFact": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "NFCorpus": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackAndroidRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "ClimateFEVER": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackEnglishRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackGamingRetrieval":  "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackGisRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackMathematicaRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackPhysicsRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackProgrammersRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackStatsRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackTexRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackUnixRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackWebmastersRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "CQADupstackWordpressRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "DBPedia": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "FEVER": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "FiQA2018": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "HotpotQA": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "MSMARCO": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "NQ": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "QuoraRetrieval": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "SCIDOCS": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "TRECCOVID": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
    "Touche2020": "<|user|>\n" \
                "Given a query and a passage, judge whether the passage is relevant to the query or not.\n\n" \
                "Query: {query}\n" \
                "Passage: {passage}\n\n" \
                "Answer with yes if the passage is relevant to the query, and no otherwise.\n" \
                "<|assistant|>\n" \
                "Answer:",
}

class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_dres_compatible(model):
        for method in DRES_METHODS:
            op = getattr(model, method, None)
            if not (callable(op)):
                return False
        return True

    def evaluate(
        self,
        model,
        split="test",
        batch_size=128,
        corpus_chunk_size=None,
        score_function="cos_sim",
        **kwargs
    ):
        task_name = kwargs['task_name']
        sgpt2_model = model
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]
        model = model if self.is_dres_compatible(model) else DRESModel(model)

        if os.getenv("RANK", None) is None:
            # Non-distributed
            from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
            model = DRES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
                **kwargs,
            )

        else:
            # Distributed (multi-GPU)
            from beir.retrieval.search.dense import (
                DenseRetrievalParallelExactSearch as DRPES,
            )
            model = DRPES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size,
                **kwargs,
            )

        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        # with open(f'qrels/{task_name}.json') as f:
        #     results = json.load(f)
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        sgpt2_model = sgpt2_model.to('cpu')
        logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        model_rerank = kwargs.get('model_rerank', None)
        template = TEMPLATES[task_name]
        if model_rerank is not None:
            if not os.path.isdir(f"rank_cache_new/{task_name}"):
                os.makedirs(f"rank_cache_new/{task_name}",exist_ok=True)
            model_rerank.tokenizer.pad_token_id = model_rerank.tokenizer.eos_token_id
            model_rerank = model_rerank.cuda()
            top_k = kwargs.get('tok_k',kwargs['top_k'])
            # step_size = kwargs.get('step_size', 2)
            # window_size = kwargs.get('window_size', -1)
            os.environ.pop("BIDIRECTIONAL_ATTN")
            print("BIDIRECTIONAL_ATTN", os.getenv("BIDIRECTIONAL_ATTN", False))
            all_qids = []
            for k in results:
                all_qids.append(k)
            for qid in all_qids:
                doc_ids = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
                # remove_doc_ids = [d[0] for d in doc_ids[top_k:]]
                # for a_doc_id in remove_doc_ids:
                #     results[qid].pop(a_doc_id)
            all_qids = []
            for k in results:
                all_qids.append(k)
            bar = tqdm(range(len(all_qids)*top_k),desc='reranking')
            def print_orders(l,tag):
                order_to_print = []
                for local_i,o in enumerate(l):
                    order_to_print.append([local_i,o])
                print(order_to_print,tag)
            for qid in all_qids:
                flag = False
                rerank_orders = {}
                if os.path.isfile(f"rank_cache_new/{task_name}/{qid}.json"):
                    # continue
                    with open(f"rank_cache_new/{task_name}/{qid}.json") as f:
                        rerank_orders = json.load(f)
                    if 'old_orders' in rerank_orders and 'new_orders' in rerank_orders:
                        flag = True
                if not flag:
                    with open(f"rank_cache_new/{task_name}/{qid}.json",'w') as f:
                        json.dump({},f,indent=2)
                    doc_ids = sorted(results[qid].items(),key=lambda x:x[1],reverse=True)
                    orders = [d[0] for d in doc_ids]
                    old_orders = copy.deepcopy(orders)
                    new_orders = []
                    for a_doc_id in orders[:top_k]:
                        # cut to both query and foc to 600 for ArguAna
                        cur_prompt = template.format(query=queries[qid][:600],passage=corpus[a_doc_id]['title']+' '+corpus[a_doc_id]['text'][:600])
                        inputs = model_rerank.tokenizer(cur_prompt, return_tensors="pt")["input_ids"].to(model_rerank.device)
                        generation_output = model_rerank.generate(inputs, max_new_tokens=1, temperature=0,
                                                                  do_sample=False, return_dict_in_generate=True,
                                                                  output_scores=True)
                        scores = generation_output.scores[0][0].cpu()
                        new_orders.append([a_doc_id,scores[5081]]) # 708 for no, 5081 for yes
                        bar.update(1)
                    new_orders_raw = sorted(new_orders,key=lambda x:x[1],reverse=True)
                    new_orders = [i[0] for i in new_orders_raw]
                    rerank_orders = {'old_orders':old_orders,'new_orders':new_orders}
                    with open(f"rank_cache_new/{task_name}/{qid}.json",'w') as f:
                        json.dump(rerank_orders,f,indent=2)
                # assert set(rerank_orders['new_orders'])==set(rerank_orders['old_orders'])
                # assert set(rerank_orders['new_orders'])==set(list(results[qid].keys()))
                # selected_scores = []
                # for rank_id,o in enumerate(rerank_orders['new_orders']):
                #     selected_scores.append(results[qid][o])
                # selected_scores = sorted(selected_scores,reverse=True)
                # for rank_id,o in enumerate(rerank_orders['new_orders']):
                #     results[qid][o] += (10-rank_id)/kwargs['divisor']
            os.environ["BIDIRECTIONAL_ATTN"] = 'true'
            print("BIDIRECTIONAL_ATTN", os.getenv("BIDIRECTIONAL_ATTN", False))

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values, ignore_identical_ids=kwargs.get("ignore_identical_ids", True))
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        print(scores)

        return scores


class DRESModel:
    """
    Dense Retrieval Exact Search (DRES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """

    def __init__(self, model, sep=" ", **kwargs):
        self.model = model
        self.sep = sep
        self.use_sbert_model = isinstance(model, SentenceTransformer)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        if self.use_sbert_model:
            if isinstance(self.model._first_module(), Transformer):
                logger.info(f"Queries will be truncated to {self.model.get_max_seq_length()} tokens.")
            elif isinstance(self.model._first_module(), WordEmbeddings):
                logger.warning(
                    "Queries will not be truncated. This could lead to memory issues. In that case please lower the batch_size."
                )
        return self.model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)
