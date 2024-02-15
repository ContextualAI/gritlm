import argparse
import json
import os
import random

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--instructions", type=str, default=None)
parser.add_argument("--data_id", type=str, default=None)
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
    'For instance, <<{}>>  to "{}"',
    'For instance, <<{}>>  to <<{}>>',
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

def diverse_instruction(instructions, task_name, is_query, data):
    num = random.random()
    # 3% chance use nothing
    if num < 0.03: return ""
    # 97% chance use one of written instructions
    else:
        # 8% chance just use generic instruction
        if num < 0.08:
            i = random.choice(GENERIC_INSTRUCTIONS)
        # 92% chance use one of written instructions
        else:
            i = random.choice(instructions).strip(". !")
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
        # 5% chance include fewshot example
        if num < 0.05:
            i += random.choice(SEPARATOR)
            fewshot = random.choice(data)
            if is_query:
                if num < 0.55:
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
        return i    

os.makedirs(args.output_dir, exist_ok=True)

data = load_dataset(args.data_id, token='YOUR_HF_KEY', cache_dir=args.cache_dir)['train']
if 'task' in data[0]:
    task_name = data[0]['task']
else:
    task_name = args.data_id.split('/')[-1]

output_path = os.path.join(args.output_dir, f"{task_name}.jsonl")
if os.path.exists(output_path):
    print(f"Output path {output_path} already exists; exiting")
    exit()

with open(args.instructions) as f:
    instructions = json.load(f)
    q_instructions = instructions.get('q_instructions', instructions.get('instructions'))
    d_instructions = instructions.get('d_instructions', instructions.get('instructions'))
    assert (q_instructions is not None) and (d_instructions is not None)

with open(output_path, 'w') as f:
    for e in data:
        # Redo instructions if already done
        if isinstance(e["query"], list):
            assert len(e["query"]) == 2
            e["query"] = e["query"][-1]
            assert all([len(x) == 2 for x in e["pos"]])
            e["pos"] = [x[-1] for x in e["pos"]]
            assert all([len(x) == 2 for x in e["neg"]])
            e["neg"] = [x[-1] for x in e["neg"]]
        f.write(json.dumps({
            'query': [diverse_instruction(q_instructions, task_name, is_query=True, data=data), e['query']],
            'pos': [[diverse_instruction(d_instructions, task_name, is_query=False, data=data), p] for p in e['pos']],
            'neg': [[diverse_instruction(d_instructions, task_name, is_query=False, data=data), p] for p in e['neg']],
        }) + '\n')
