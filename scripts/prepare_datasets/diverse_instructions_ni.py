import argparse
import json
import itertools
import os
import random

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--instructions", type=str, default=None)
parser.add_argument("--data_id", type=str, default=None,choices=["multi-train/natural_instruction_dataset1", "multi-train/natural_instruction_dataset2", "multi-train/natural_instruction_dataset3", "multi-train/natural_instruction_dataset3"])
parser.add_argument("--cache_dir", type=str, default=None)
args = parser.parse_args()

GENERIC_INSTRUCTIONS = [
    "Represent this text",
    "Represent this",
    "Represent the input",
    "Represent the following document",
    "Represent the natural language",
    "Represent",
    "Represent the next text",
    "Represent text",
]

TASK_INSTRUCTIONS_QUERY = [
    "{}\n\nGiven the above task description, represent the following input that will be used to find an answer",
    "Below is a task description. Afterwards, you will need to represent an example of that task that is to be matched with a correct answer.\n\n{}",
    "The task is: {}\n\nRepresent the following input to retrieve a fitting output",
    "{}\n\n\nRepresent the following input to retrieve a matching output",
    "For the following task, we need to match inputs with outputs via retrieval. After the task description, you will be given an example input for represenation.\n\nTask: {}",
    "Information: {}\nRepresent the text to be matched with a correct answer for it given the information",
]

TASK_INSTRUCTIONS_DOC = [
    "{}\n\nGiven the above task description, represent the following answer that will be used to retrieve a fitting input",
    "Below is a task description. Afterwards, you will need to represent an example answer of that task that is to be matched with a fitting input.\n\n{}",
    "The task is: {}\n\nRepresent the following output to retrieve a fitting input",
    "{}\n\n\nRepresent the following output to retrieve a matching input",
    "For the following task, we need to match inputs with outputs via retrieval. After the task description, you will be given an example output for represenation.\n\nTask: {}",
    "Information: {}\nRepresent the text to be matched with a fitting input for it given the information",
]

TRIPLET_FEWSHOT_INSTRUCTIONS = [
    "For instance you may be given '{}' and it should match with '{}' but not with '{}'.",
    "E.g. given '{}' it should be close to '{}' but not to '{}'.",
    "For example, '{}' should have a representation like '{}' but very far from '{}'.",
    "E.g. {} == {} != {}",
    "E.g. '{}' == '{}' != '{}'",
    'E.g. "{}" == "{}" != "{}"',
    "E.g.\n{} == {} != {}",
    "E.g.\n'{}' == '{}' != '{}'",
    'E.g.\n"{}" == "{}" != "{}"',
    "Examples:\n\nGiven {} it matches with {} but not with {}",
    "Examples:\nProvided: {} Match: {} Hard Negative: {}",
    "Given {}, a positive would be {} & a negative would be {}",
    "The provided query could be '{}' and the positive '{}' and the negative '{}'",
    "The query could be '{}' and should be close to '{}' but very far from '{}'",
    "Examples:\n'{}' == '{}' != '{}'",
    "Fewshots:\n'{}' == '{}' != '{}'",    
]
PAIR_FEWSHOT_INSTRUCTIONS = [
    "For example, '{}' should be close to '{}'",
    "For example, {} should be similar to {}",
    'To give you a sense - "{}" should be close to "{}"',
    'For instance, <<{}>> to "{}"',
    'For instance, <<{}>> to <<{}>>',
    'Examples:\n\n"{}" == "{}"',
    'Example:\nProvided: "{}" Match: "{}"',
    'Given {}, a positive would be {}',
    'The provided query could be "{}" and the positive "{}"',
    'E.g. {} == {}',
    'Fewshot example: "{}" == "{}"',
    'E.g.:\n{} == {}',
    'Examples:\n\n\n"{}" == "{}"',
    'Examples:\nProvided: "{}" Match: "{}"',
]
SEPARATOR = [
    " ",
    "\n\n",
    "\n",
    "\n------\n",
    "\n\n------\n\n",
    "\n",
    "\n\n",
    "\n\n\n",
    "\n\n\n",
    "\n",
]

random.seed(42)

def diverse_instruction(meta, is_query, data):
    num = random.random()
    # 0.1% chance use nothing
    if num < 0.001: return ""
    # 99.9% chance use one of written instructions
    else:
        # 0.5% chance just use generic instruction
        if num < 0.006:
            i = random.choice(GENERIC_INSTRUCTIONS)
        # 99.4% chance use one of written instructions
        else:
            # Usually always list of 1 item except for task288_gigaword_summarization with 2 options
            definition = random.choice(meta['Definition'])
            if is_query:
                i = random.choice(TASK_INSTRUCTIONS_QUERY).format(definition)
            else:
                i = random.choice(TASK_INSTRUCTIONS_DOC).format(definition)
        num = random.random()
        # 10% chance add "."
        if (num < 0.1) and (i[-1] not in ["?"]):
            i += "."
        # 5% chance add "!"
        elif num < 0.15:
            i += "!"
        # 5% chance add ":"
        elif num < 0.20:
            i += ":"
        # 20% chance make all lower caps
        num = random.random()
        if num < 0.2:
            i = i.lower()
        # 10% chance include fewshot example
        if num < 0.1:
            i += random.choice(SEPARATOR)
            fewshot = random.choice(data)
            if is_query:
                if num < 0.60:
                    i += random.choice(TRIPLET_FEWSHOT_INSTRUCTIONS).format(
                        fewshot['query'], 
                        random.choice(fewshot['pos']), 
                        random.choice(fewshot['neg']),
                    )
                else:
                    i += random.choice(PAIR_FEWSHOT_INSTRUCTIONS).format(
                        fewshot['query'], 
                        random.choice(fewshot['pos'])
                    )
            else:
                # Reverse as doc will be given
                i += random.choice(PAIR_FEWSHOT_INSTRUCTIONS).format(
                    random.choice(fewshot['pos']),
                    fewshot['query']
                )
        num = random.random()
        # 2.5% chance include domain
        if num < 0.025:
            i = f"Domains: {','.join(meta['Domains'])}" + random.choice(SEPARATOR) + i
        num = random.random()
        # 2.5% chance include category
        if num < 0.025:
            i = f"Categories: {','.join(meta['Categories'])}" + random.choice(SEPARATOR) + i        
        return i    

os.makedirs(args.output_dir, exist_ok=True)

data = load_dataset(args.data_id, token='YOUR_TOKEN', cache_dir=args.cache_dir)['train']

with open("/home/niklas/gritlm/scripts/prepare_datasets/natural_instruction_dataset_metadata.json") as f:
    metadata = json.load(f)

for task_id, dg in itertools.groupby(data, key=lambda x: x["task"]):
    print("Running:", task_id)
    meta = metadata[task_id]
    dg = list(dg)
    path = os.path.join(args.output_dir, f"{task_id}.jsonl")
    with open(path, 'a') as f:
        for e in dg:
            f.write(json.dumps({
                'query': [diverse_instruction(meta, is_query=True, data=dg), e['query']],
                'pos': [[diverse_instruction(meta, is_query=False, data=dg), p] for p in e['pos']],
                'neg': [[diverse_instruction(meta, is_query=False, data=dg), p] for p in e['neg']],
            }) + '\n')
