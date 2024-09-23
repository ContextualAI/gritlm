import json
import logging
import math
import os
import pickle
from typing import Optional, Set, Tuple, Union, Any
import numpy as np
import torch

from rag import dist_utils

logger = logging.getLogger(__name__)

DTYPE_TO_TORCH_DTYPE = {
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float16': torch.float16,
}


def serialize_listdocs(ids):
    ids = pickle.dumps(ids)
    ids = torch.tensor(list(ids), dtype=torch.uint8).cuda()
    return ids


def deserialize_listdocs(ids):
    return [pickle.loads(x.cpu().numpy().tobytes()) for x in ids]


class DistributedIndex(object):
    def __init__(self, dtype=torch.float32):
        self.embeddings = None
        self.doc_map = dict()
        self.is_in_gpu = True if torch.cuda.is_available() else False
        self.dtype = dtype

    def init_embeddings(self, passages, dim: Optional[int]):
        self.doc_map = {i: doc for i, doc in enumerate(passages)}
        self.embeddings = torch.zeros(dim, (len(passages)), dtype=self.dtype)
        if self.is_in_gpu:
            self.embeddings = self.embeddings.cuda()

    def _get_saved_embedding_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"embeddings.{shard}.pt")

    def _get_saved_passages_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"passages.{shard}.pt")

    def save_index(self, path: str, total_saved_shards: int, overwrite_saved_passages: bool = False) -> None:
        """
        Saves index state to disk, which can later be loaded by the load_index method.
        Specifically, it saves the embeddings and passages into total_saved_shards separate file shards.
        This option enables loading the index in another session with a different number of workers, as long as the number of workers is divisible by total_saved_shards.
        Note that the embeddings will always be saved to disk (it will overwrite any embeddings previously saved there).
        The passages will only be saved to disk if they have not already been written to the save directory before, unless the option --overwrite_saved_passages is passed.
        """
        assert self.embeddings is not None
        rank = dist_utils.get_rank()
        ws = dist_utils.get_world_size()
        assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        n_embeddings = self.embeddings.shape[1]
        embeddings_per_shard = math.ceil(n_embeddings / shards_per_worker)
        assert n_embeddings == len(self.doc_map), len(self.doc_map)
        for shard_ind, (shard_start) in enumerate(range(0, n_embeddings, embeddings_per_shard)):
            shard_end = min(shard_start + embeddings_per_shard, n_embeddings)
            shard_id = shard_ind + rank * shards_per_worker  # get global shard number
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            if not os.path.exists(passage_shard_path) or overwrite_saved_passages:
                passage_shard = [self.doc_map[i] for i in range(shard_start, shard_end)]
                with open(passage_shard_path, "wb") as fobj:
                    pickle.dump(passage_shard, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            embeddings_shard = self.embeddings[:, shard_start:shard_end]#.clone()
            embedding_shard_path = self._get_saved_embedding_path(path, shard_id)
            torch.save(embeddings_shard, embedding_shard_path)

    def load_index(self, path: str, total_saved_shards: int):
        """
        Loads sharded embeddings and passages files (no index is loaded).
        """
        rank = dist_utils.get_rank()
        ws = dist_utils.get_world_size()
        assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        passages = []
        embeddings = []
        for shard_id in range(rank * shards_per_worker, (rank + 1) * shards_per_worker):
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            with open(passage_shard_path, "rb") as fobj:
                passages.append(pickle.load(fobj))
            embeddings_shard_path = self._get_saved_embedding_path(path, shard_id)
            if self.is_in_gpu:
                embeddings.append(torch.load(embeddings_shard_path, map_location="cpu").cuda())
            else:
                embeddings.append(torch.load(embeddings_shard_path, map_location="cpu"))
        self.doc_map = {}
        n_passages = 0
        for chunk in passages:
            for p in chunk:
                self.doc_map[n_passages] = p
                n_passages += 1
        if len(embeddings) > 1:
            self.embeddings = torch.concat(embeddings, dim=1)
        else:
            self.embeddings = embeddings[0]

    def _compute_scores_and_indices(self, allqueries: torch.tensor, topk: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the distance matrix for the query embeddings and embeddings chunk and returns the k-nearest neighbours and corresponding scores.
        """
        scores = torch.matmul(allqueries.to(self.embeddings.device), self.embeddings)
        scores, indices = torch.topk(scores, topk, dim=1)

        return scores, indices

    @torch.no_grad()
    def search_knn(self, queries, topk):
        """
        Conducts exhaustive search of the k-nearest neighbours using the inner product metric.
        """
        allqueries = dist_utils.varsize_all_gather(queries)
        allsizes = dist_utils.get_varsize(queries)
        allsizes = np.cumsum([0] + allsizes.cpu().tolist())
        # compute scores for the part of the index located on each process
        scores, indices = self._compute_scores_and_indices(allqueries, topk)
        indices = indices.tolist()
        docs = [[self.doc_map[x] for x in sample_indices] for sample_indices in indices]
        if torch.distributed.is_initialized():
            docs = [docs[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            docs = [serialize_listdocs(x) for x in docs]
            scores = [scores[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            gather_docs = [dist_utils.varsize_gather(docs[k], dst=k, dim=0) for k in range(dist_utils.get_world_size())]
            gather_scores = [
                dist_utils.varsize_gather(scores[k], dst=k, dim=1) for k in range(dist_utils.get_world_size())
            ]
            rank_scores = gather_scores[dist_utils.get_rank()]
            rank_docs = gather_docs[dist_utils.get_rank()]
            scores = torch.cat(rank_scores, dim=1)
            rank_docs = deserialize_listdocs(rank_docs)
            merge_docs = [[] for _ in range(queries.size(0))]
            for docs in rank_docs:
                for k, x in enumerate(docs):
                    merge_docs[k].extend(x)
            docs = merge_docs
        _, subindices = torch.topk(scores, topk, dim=1)
        scores = scores.tolist()
        subindices = subindices.tolist()
        # Extract topk scores and associated ids
        scores = [[scores[k][j] for j in idx] for k, idx in enumerate(subindices)]
        docs = [[docs[k][j] for j in idx] for k, idx in enumerate(subindices)]
        return docs, scores

    def is_index_trained(self) -> bool:
        return True


def load_passages(filenames, maxload=-1):
    def process_jsonl(
        fname,
        counter,
        passages,
        world_size,
        global_rank,
        maxload,
    ):
        def load_item(line):
            if line.strip() != "":
                item = json.loads(line)
                if "title" in item and "section" in item and len(item["section"]) > 0:
                    item["title"] = f"{item['title']}: {item['section']}"
                return item
            else:
                print("empty line")

        for line in open(fname):
            if maxload > -1 and counter >= maxload:
                break

            ex = None
            if (counter % world_size) == global_rank:
                ex = load_item(line)
                passages.append(ex)
            counter += 1
        return passages, counter

    counter = 0
    passages = []
    global_rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    for filename in filenames:

        passages, counter = process_jsonl(
            filename,
            counter,
            passages,
            world_size,
            global_rank,
            maxload,
        )

    return passages

def load_or_initialize_index(args, logger, dim):
    index = DistributedIndex(dtype=DTYPE_TO_TORCH_DTYPE[args.idxdtype])

    if args.load_index_path is not None:
        logger.info(f"Loading index from: {args.load_index_path}")
        index.load_index(args.load_index_path, args.save_index_n_shards)
        passages = [index.doc_map[i] for i in range(len(index.doc_map))]
    else:
        logger.info(f"Loading passages from: {args.passages}")
        passages = load_passages(args.passages)
        logger.info(f"Loaded {len(passages)} passages")
        if args.limit is not None:            
            passages = passages[args.limit_start:args.limit]
            #passages = [max(passages, key=lambda x: len(x["text"]))]
            logger.info(f"Limiting to {len(passages)} passages ({args.limit_start}-{args.limit})")
        if args.customd:
            if os.path.exists(args.customd):
                with open(args.customd, "r") as f:
                    passages = [{"text": f.read(), "title": ""}]
            else:
                # Is number
                passages = [{"text": "<s>" * int(args.customd), "title": ""}]
        logger.info(f"Example passage: {passages[0]}")
        index.init_embeddings(passages, dim)

    return index, passages
