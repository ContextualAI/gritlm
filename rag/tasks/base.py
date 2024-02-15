import json
import logging
import random
from collections import defaultdict

from rag.tasks.evaluation import exact_match_score

logger = logging.getLogger(__name__)


class BaseTask(object):
    metrics = ["accuracy", "eval_loss"]

    def __init__(self, *args, **kwargs):
        self.filter = None

    @staticmethod
    def data_iterator(filenames, world_rank=-1, world_size=-1, repeat_if_less_than_world_size=False, *args, **kwargs):
        if isinstance(filenames, str):
            filenames = [filenames]

        def _iter():
            # iterate over files
            return (line for filename in filenames for line in open(filename, encoding="utf-8"))

        def _stop():
            # stop iterating over data when at least one example has been fed to each worker
            return (total_yielded >= world_size) if repeat_if_less_than_world_size else (total_yielded > 0)

        total_yielded = 0
        while not _stop():
            for line in _iter():
                total_yielded += 1
                if world_rank > -1 and total_yielded % world_size != world_rank:
                    continue
                example = json.loads(line)
                yield example

    @staticmethod
    def batch_iterator(data_iterator, batch_size, drop_last=False, shuffle=False):
        if shuffle:
            data_iterator = BaseTask.shuffle_iterator(data_iterator)
        batch = defaultdict(lambda: [])
        batch["__size__"] = 0
        batch_counter = 0
        for example in data_iterator:
            for k, v in example.items():
                batch[k].append(v)
            batch["__size__"] += 1
            if batch["__size__"] == batch_size:
                batch_counter += 1
                yield batch
                batch = defaultdict(lambda: [])
                batch["__size__"] = 0
        if batch["__size__"] > 0 and not drop_last:
            yield batch

    def evaluation(self, prediction, ground_truths):
        """most basic evaluation: checks if prediction matches ground truth"""
        sample_metrics = {"accuracy": exact_match_score(prediction, ground_truths)}
        return sample_metrics

    @staticmethod
    def shuffle_iterator(dataset):
        d = list(dataset)
        random.shuffle(d)
        for x in d:
            yield x

    def process(self, example, *args, **kwargs):
        """most basic example processing, should be overwritten in subclasses"""
        assert "target" in example, "base task requires a `target` field string to be defined"
        assert "query" in example, "base task requires a `query` field string to be defined"
        assert type(example["target"]) == str, "base task requires a `target` field string to be defined"
        assert type(example["query"]) == str, "base task requires a `query` field string to be defined"

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        return example

    def evaluation_postprocessing(self, metrics, dataset_with_predictions):
        """do any necessary postprocessing of generated predictions or metrics after the evaluation loop"""
        return metrics, dataset_with_predictions


def filter_results_by_id(batch_metadata, passages, scores, topk, training=False):
    """
    Removes retrieved passages from retrieved set if their id is the same as the instance in the batch metadata.
    Useful for MLM or LM where we dont want model to "cheat" by retrieving the passgage it is denoising/generating.

    If, once violating passages are removed, there are < topk results, the violating passages will be added back,
    in with a warning
    """

    if batch_metadata is None:
        logger.warning("Trying to filter a batch with no metadata - probably a padding instance - just return the topk")
        return [ps[:topk] for ps in passages], [ss[:topk] for ss in scores]

    def _same_passage_chunk(source_metadata, passage):
        return passage["id"] == source_metadata["id"]

    output_passages, output_scores = [], []

    for metadata, passage_li, scores_li in zip(batch_metadata, passages, scores):

        filtered_passages_and_scores, violating_passages_and_scores = [], []
        for (p, s) in zip(passage_li, scores_li):
            if not _same_passage_chunk(metadata, p):
                filtered_passages_and_scores.append((p, s))
            else:
                violating_passages_and_scores.append((p, s))

        if topk > len(filtered_passages_and_scores):
            logger.warning(f"{len(filtered_passages_and_scores)} passages after filtering for topk = {topk}")

        filtered_passages_and_scores += violating_passages_and_scores
        filtered_passages, filtered_scores = zip(*filtered_passages_and_scores)
        output_passages.append(filtered_passages)
        output_scores.append(filtered_scores)

    return [ps[:topk] for ps in output_passages], [ss[:topk] for ss in output_scores]