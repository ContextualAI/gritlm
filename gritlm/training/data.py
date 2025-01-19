from dataclasses import dataclass
import logging
import math
import random
from typing import Iterator, List, Tuple, Union

import datasets
import torch
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments

logger = logging.getLogger(__name__)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Union[datasets.Dataset, List[datasets.Dataset]],
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        mode: str = 'embedding',
        full_bs: int = None,
        generative_bs: int = None,
        max_seq_len: int = 2048,
    ):
        self.indices_emb, self.indices_gen = None, None
        if mode == 'unified':
            self.ds_embedding = dataset[0]
            self.ds_generative = dataset[1]
            self.len_embedding = len(self.ds_embedding)
            self.len_generative = len(self.ds_generative)
            self.total_len = max(self.len_embedding, self.len_generative)
            if args.use_unique_indices: self.set_indices()
        elif mode == 'embedding': 
            self.ds_embedding = dataset
            self.total_len = self.len_embedding = len(self.ds_embedding)
        elif mode == 'generative': 
            self.ds_generative = dataset
            self.total_len = self.len_generative = len(self.ds_generative)
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        # Too long items will be stuck in communication so cut them on the fly
        self.max_char_len = max_seq_len * 10

        self.n_samples = self.total_len * full_bs
        if generative_bs is not None:
            assert full_bs >= generative_bs, "Full batch size must be larger than generative batch size"
            assert full_bs % generative_bs == 0, "Full batch size must be divisible by generative batch size"
            self.take_nth = full_bs // generative_bs
        else:
            self.take_nth = 1

    def set_indices(self):
        """
        When embedding/generative datasets are of different sizes, ensure that the smaller dataset is still
        randomly sampled from even though the __getitem__ idx may be out of range as it is for the bigger one.
        Do so by maintaining a set of indices to sample from which are unique for each process.
        """
        if self.len_embedding > self.len_generative:
            indices_gen = list(range(self.len_generative))
            if torch.distributed.is_initialized():
                # world_size and rank are global (i.e. across all nodes and processes)
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                indices_gen = indices_gen[rank::world_size]
            self.indices_gen = set(indices_gen)
        elif self.len_embedding < self.len_generative:
            indices_emb = list(range(self.len_embedding))
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                indices_emb = indices_emb[rank::world_size]
            self.indices_emb = set(indices_emb)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], BatchEncoding]:
        """
        Problems:
        If training for >1 epoch in unified mode, the same generative & embedding samples will 
        always be in the same batch as the same index is used for both datasets.
        Solution:
        Don't train for >1 epoch by duplicating the dataset you want to repeat in the folder.
        Upon loading, each dataset is shuffled so indices will be different.
        """
        query, passages, generative = None, None, None
        if self.mode in ["unified", "embedding"]:
            if self.indices_emb is not None:
                if not self.indices_emb:
                    self.set_indices()
                item = self.indices_emb.pop()
            elif item >= self.len_embedding:
                item = random.randint(0, self.len_embedding - 1)
            query = self.ds_embedding[item]['query']

            if isinstance(query, str):
                query = query[:self.max_char_len]
            elif isinstance(query, list):
                query = [x[:self.max_char_len] for x in query]
            
            passages = []
            pos = random.choice(self.ds_embedding[item]['pos'])

            if isinstance(pos, str):
                pos = pos[:self.max_char_len]
            elif isinstance(pos, list):
                pos = [x[:self.max_char_len] for x in pos]
            else:
                raise ValueError(f"Unexpected type for pos: {type(pos)}")
            passages.append(pos)

            if len(self.ds_embedding[item]['neg']) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(self.ds_embedding[item]['neg']))
                negs = random.sample(self.ds_embedding[item]['neg'] * num, self.args.train_group_size - 1)
            else:
                negs = random.sample(self.ds_embedding[item]['neg'], self.args.train_group_size - 1)
            
            for i, neg in enumerate(negs):
                if isinstance(neg, str):
                    negs[i] = neg[:self.max_char_len]
                elif isinstance(neg, list):
                    negs[i] = [x[:self.max_char_len] for x in neg]
                else:
                    raise ValueError(f"Unexpected type for neg: {type(neg)}")
            passages.extend(negs)

        if (self.mode in ["unified", "generative"]) and (self.n_samples % self.take_nth == 0):
            if self.indices_gen is not None:
                if not self.indices_gen:
                    self.set_indices()
                item = self.indices_gen.pop()
            elif item >= self.len_generative:
                item = random.randint(0, self.len_generative - 1)
            generative = self.ds_generative[item]["text"]

        self.n_samples -= 1
        return query, passages, generative

@dataclass
class CustomCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    generative_max_len: int = 128

    base_bos: str = ""
    turn_sep: str = ""

    user_bos: str = ""
    user_eos: str = ""

    embed_bos: str = ""
    # Am embed eos is useless as there is no generative loss on it so it won't be learned
    # & it does not add anything new; It only makes sense for lasttoken pooling
    embed_eos: str = ""

    assistant_bos: str = ""
    assistant_eos: str = ""

    prefixlm: bool = False

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        generative = [f[2] for f in features]

        # Flatten if list of lists
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        features = {}

        # If each sample is a tuple it is of format (instruction, text)
        q_instruction_lens, g_instruction_lens = None, None
        if isinstance(query[0], (tuple, list)):
            q_instruction_lens = [
                len(self.tokenizer.tokenize(
                    self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos
                    if f[0].strip("\t\n :") else self.base_bos + self.embed_bos.lstrip()
                )) for f in query
            ]
            d_instruction_lens = [
                len(self.tokenizer.tokenize(
                    self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos
                    if f[0].strip("\t\n :") else self.base_bos + self.embed_bos.lstrip()
                )) for f in passage
            ]

            # Strip including `:` which is added in MEDI but no longer needed due to the format with special tokens
            query = [
                self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos + f[1] + self.embed_eos
                if f[0].strip("\t\n :") else self.base_bos + self.embed_bos.lstrip() + f[1] + self.embed_eos for f in query
            ]
            passage = [
                self.base_bos + self.user_bos + f[0].strip("\t\n :") + self.user_eos + self.embed_bos + f[1] + self.embed_eos
                if f[0].strip("\t\n :") else self.base_bos + self.embed_bos.lstrip() + f[1] + self.embed_eos for f in passage
            ]

        # If each sample is a tuple it is of format (instruction, text, instruction, text, ...)
        if isinstance(generative[0], (tuple, list)):
            # Do not strip user input as model should be robust to it
            # Need to check for None in case gen batch size is smaller than emb batch size
            # instruction and text are a list each, as it may be multi-turn
            g_instruction_lens = [
                [
                    len(
                        self.tokenizer.tokenize(self.user_bos + z + self.user_eos + self.assistant_bos)
                        if i > 0 else
                        self.tokenizer.tokenize(self.base_bos + self.user_bos + z + self.user_eos + self.assistant_bos)
                    )
                    if i % 2 == 0 else
                    len(self.tokenizer.tokenize(z.strip() + self.assistant_eos))
                    for i, z in enumerate(f[:-1])
                ] for f in generative if f is not None
            ]
            generative = [
                self.base_bos + self.turn_sep.join([
                    self.user_bos + f[i] + self.user_eos + self.assistant_bos + f[i+1].strip() + self.assistant_eos for i in range(0, len(f), 2)
                ]) for f in generative if f is not None
            ]

        if query[0] is not None:
            features["query"] = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
                add_special_tokens=False, # BOS / EOS is already in the prompt
            )
            features["passage"] = self.tokenizer(
                passage,
                padding=True,
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
                add_special_tokens=False, # BOS / EOS is already in the prompt
            )

        if generative[0] is not None:
            features["generative"] = self.tokenizer(
                generative,
                padding=True,
                truncation=True,
                max_length=self.generative_max_len,
                return_tensors="pt",
                add_special_tokens=False, # BOS / EOS is already in the prompt
            )
            features["generative"]["labels"] = features["generative"]["input_ids"].clone()
            # Do not mask out the first token as it is always something & could be the pad token id (bos)
            features["generative"]["labels"][:,1:][features["generative"]["labels"][:,1:] == self.tokenizer.pad_token_id] = -100

        if q_instruction_lens:
            # Check that there is no mistake
            for i, l in enumerate(q_instruction_lens):
                assert features["query"]["input_ids"][i, l] != self.tokenizer.pad_token, f"No text to embed: {query[i]}"
            for i, l in enumerate(d_instruction_lens):
                assert features["passage"]["input_ids"][i, l] != self.tokenizer.pad_token, f"No text to embed: {passage[i]}"
            # Need to be masked out later
            features["query"]["instruction_lens"] = torch.tensor(q_instruction_lens)
            features["passage"]["instruction_lens"] = torch.tensor(d_instruction_lens)
        if g_instruction_lens:
            # Mask instructions as -100 to be ignored in the loss
            # If multiturn, instructions are masked in multiple places
            for i, lengths in enumerate(g_instruction_lens):
                cur_len = 0
                for j, l in enumerate(lengths):
                    # For PrefixLM mask everything up to latest assistant utterance
                    if (j % 2 == 0) or self.prefixlm:
                        features["generative"]["labels"][i, cur_len:cur_len+l] = -100
                    cur_len += l

        return features

@dataclass
class CustomRandomSampler(torch.utils.data.sampler.RandomSampler):
    """
    Sampler used when training on multiple datasets to ensure each 
    batch only contains samples from one dataset for the majority of cases.
    """
    total_batch_size: int = 8
    ds_lens: List[int] = None
    _num_samples: int = None
    data_source: CustomDataset = None
    replacement: bool = False

    def __iter__(self) -> Iterator[int]:
        
        if not hasattr(self, "generator") or self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # We have multiple datasets each with a different number of samples
        # e.g. [100, 150, 50]
        # We would like to sample from them such that as much as possible each batch
        # only has samples from the same dataset.
        # For example if our batch size is 4 then
        # indices might be [0,1,2,3,100,101,102,103,150,151,152,153,50,51,52,53]
        # To do so:
        # 1. Shuffle the indices of each dataset separately
        # 2. Create batches with only samples from one dataset
        # 3. Keep the remaining samples which do not fit into a batch separate
        # 4. Then create mixed batches from the remaining samples
        # 5. Then yield randomly from all the batches
        # Testing:
        # ds_lens = [100, 150, 50]
        # batch_size = 8
        # Create random indices for each dataset
        ds_indices = [torch.randperm(n, generator=generator).tolist() for n in self.ds_lens]
        # Increase the indices to be indices of the concatenated dataset
        ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
        # Create batches with only samples from one dataset
        ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in ds_batches:
            if len(b[-1]) < self.total_batch_size:
                incomplete_indices.append(b.pop())

        if incomplete_indices:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=generator).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(torch.tensor(incomplete_indices), self.total_batch_size))
            if len(mixed_batches[-1]) < self.total_batch_size:
                mixed_batches.pop()
            # Merge all batches to look like [...tensor([259, 273, 284, 289]), tensor([262, 280, 295, 258]), ...]
            ds_batches = sum(ds_batches, []) + mixed_batches
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches) - len(mixed_batches)} single-dataset batches & {len(mixed_batches)} mixed dataset batches.")
        else:
            ds_batches = sum(ds_batches, [])
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches)} single-dataset batches.")

        # Randomly permute the order of all batches, then merge them to look like tensor([...259, 273, 284, 289, 262, 280, 295, 258...])
        order = torch.randperm(len(ds_batches), generator=generator).tolist()
        ds_batches = [int(i) for i in torch.cat([ds_batches[i] for i in order]).tolist()]
        # Yield the indices
        yield from ds_batches
