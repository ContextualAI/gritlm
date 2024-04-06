"""
To reproduce Reranking experiments with using GritLM for embedding and subsequent reranking replace the AbsTaskRetrieval file in MTEB with this file.
"""
import json
import logging
from time import time
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings
import os

from .AbsTask import AbsTask
from tqdm import tqdm

logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]

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
        # if os.path.isfile('SciFact/SciFact.json'):
        #     with open('SciFact/SciFact.json') as f:
        #         results = json.load(f)
        # else:
        results = retriever.retrieve(corpus, queries)
            # with open('SciFact/SciFact.json','w') as f:
            #     json.dump(results,f,indent=2)
        end_time = time()
        sgpt2_model = sgpt2_model.to('cpu')
        logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        model_rerank = kwargs.get('model_rerank', None)
        template = "<|user|>\nI will provide you with {num} passages, each indicated by a numerical identifier []. " \
                   "Rank the passages based on their relevance to the search query {query}.\n\n{passages}\n\n" \
                   "Search Query: {query}.\n\n" \
                   "Rank the {num} passages above based on their relevance to the search query. All the passages " \
                   "should be included and listed using identifiers, in descending order of relevance. " \
                   "The output format should be [] > [] > ..., e.g., [4] > [2] > ... " \
                   "Only respond with the ranking results, do not say any word or explain.\n<|assistant|>\n"
        if model_rerank is not None:
            model_rerank = model_rerank.cuda()
            os.environ.pop("BIDIRECTIONAL_ATTN")
            print("BIDIRECTIONAL_ATTN", os.getenv("BIDIRECTIONAL_ATTN", False))
            for qid, doc_ids in tqdm(results.items(), desc='reranking'):
                # if os.path.isfile(f"SciFact/{qid}.json"):
                #     with open(f"SciFact/{qid}.json") as f:
                #         rerank_orders = json.load(f)
                # else:
                doc_ids = sorted(doc_ids.items(),key=lambda x:x[1],reverse=True)
                cur_query = queries[qid]
                num = 0
                passages = ''
                cur_prompt = None
                all_ids = {}
                scores = []
                old_orders = []
                while len(model_rerank.tokenizer(template.format(num=num, query=cur_query, passages=passages), return_tensors="pt")["input_ids"][0])<1900:
                    cur_prompt = template.format(num=num, query=cur_query, passages=passages)
                    passages += f"[{num}] {corpus[doc_ids[num][0]]['title'] + ' ' + corpus[doc_ids[num][0]]['text']}\n"
                    old_orders.append(doc_ids[num][0])
                    all_ids[num] = doc_ids[num][0]
                    scores.append(doc_ids[num][1])
                    num += 1
                inputs = model_rerank.tokenizer(cur_prompt, return_tensors="pt")["input_ids"].to(model_rerank.device)
                generation_output = model_rerank.generate(inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
                outputs = model_rerank.tokenizer.batch_decode(generation_output[:, inputs.shape[-1]:])[0].strip('</s>').strip()
                components = outputs.split('>')
                new_orders = []
                for idx,c in enumerate(components):
                    try:
                        new_orders.append(all_ids[int(c.strip().strip('[').strip(']').strip())])
                    except:
                        print(len(old_orders),outputs)
                        pass
                rerank_orders = {'old_orders':old_orders,'new_orders':new_orders}
                    # with open(f"SciFact/{qid}.json",'w') as f:
                    #     json.dump(rerank_orders,f,indent=2)
                cur_scores = []
                for i in rerank_orders['old_orders']:
                    cur_scores.append(results[qid][i])
                for i,s in zip(rerank_orders['new_orders'],cur_scores):
                    results[qid][i] = s
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
