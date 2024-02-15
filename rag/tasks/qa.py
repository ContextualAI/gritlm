import random

from rag.tasks.evaluation import exact_match_score, match_score, f1_score, normalize_answer
from rag.tasks.base import BaseTask


class Task(BaseTask):
    metrics = ["exact_match", "match", "f1", "eval_loss"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_qa_prompt(self, question: str) -> str:
        return self.qa_prompt_format_str.format(question=question)

    def process(self, example, *args, **kwargs):

        if "target" in example:
            target = example["target"]
        elif "answers" in example:
            target = random.choice(example["answers"])
        else:
            target = None

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["metadata"] = example.get("metadata", {})
        example["query"] = example["question"]
        if target is not None:
            example["target"] = target

        return example

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "match": match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics