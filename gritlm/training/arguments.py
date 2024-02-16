from dataclasses import dataclass, field
import os
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    pooling_method: str = field(default='weightedmean', metadata={"help": "Pooling method for sentences"})
    normalized: bool = field(default=True)
    attn_implementation: str = field(default='sdpa', metadata={"help": "eager/sdpa/flash_attention_2"})
    attn: str = field(
        default='bbcc',
        metadata={
            "help": "bidirectional/causal attn for emb inst., emb sample, gen inst., gen sample"
                    " e.g. bbcc is bidirectional over both emb inst. & sample but causal over gen inst. & sample"
                    " cccc is causal over all; bccc is bidirectional over emb inst. but causal over rest etc."
        }
    )
    projection: int = field(default=None, metadata={"help": "Optional linear learned embedding down projection"})

@dataclass
class DataArguments:
    train_data: str = field(
        default=None,
        metadata={
            "help": "Path to folder or file with training data. For toy data in instruction format"
                    " point to `toy_data_instruct` instead. If the path is a folder, for each minibatch"
                    " all samples will come from one file in the folder. You can use this to ensure"
                    " in-batch negatives are very difficult."
        }
    )
    train_group_size: int = field(
        default=2,
        metadata={
            "help": "Number of positive & negatives for a query in training. There is always one"
                    " positive, so this argument controls the number of negatives" 
                    " (#negatives=train_group_size-1). Note that the number of negatives should"
                    " not be larger than the numbers of negatives in the data. Besides the negatives" 
                    " in this group, the in-batch negatives will also be used in fine-tuning."
            }
        )
    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum tokens for the query. Sequences longer"
                    " than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum tokens for passages (positives & negatives). Sequences longer"
                    " than this will be truncated, sequences shorter will be padded."
        },
    )
    generative_max_len: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization for generative. Sequences longer"
                    " than this will be truncated, sequences shorter will be padded. Defaults to --passage_max_len"
        },
    )
    max_example_num_per_dataset: int = field(
        default=100_000_000, metadata={"help": "the max number of examples for each dataset"}
    )
    num_samples: Optional[str] = field(
        default=None, metadata={"help": "path to json with number of samples per dataset"}
    )    
    use_unique_indices: bool = field(
        default=False, 
        metadata={"help": "If unified with different emb & gen dataset lens, ensure samples are unique in each epoch"}
    )
    prefixlm: bool = field(default=False, metadata={"help": "PrefixLM for generative"})


    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class CustomTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(
        default=False, 
        metadata={
            "help": "Share the negatives across all GPUs. This argument will extend the number of negatives."
        }
    )
    temperature: Optional[float] = field(
        default=0.02,
        metadata={
            "help": "Similarity will be sim = sim/temperature before using them to compute loss."
            " A higher temperature can reduce the value of similarity between texts in downstream tasks."
        }
    )
    mode: str = field(
        default='embedding', 
        metadata={
            "help": "One of ['unified', 'embedding', 'generative']. For unified,"
            " `train_data` should point to a folder with both embedding and generative data."
        }
    )
    per_device_generative_bs: int = field(
        default=None, 
        metadata={
            "help": "Per device generative batch size. It has to be smaller than the regular batch size."
                    " It will overwrite the gradient accumulation steps for generative."
                    " It only makes sense to use this argument when doing unified mode."
        }
    )
    no_gen_gas: bool = field(
        default=False, 
        metadata={
            "help": "Do not use gradient accumulation steps for generative."
            " Using both `no_gen_gas` and `no_emb_gas` will activate the GradCache Trainer"
            " but without gradient accumulation. This is useful for saving memory as it"
            " computes the backward for embedding first and then the backward for generative"
            " which is more memory efficient than the default of computing both losses,"
            " adding them and then doing the backward (https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch)."
        }
    )
    no_emb_gas: bool = field(
        default=False, 
        metadata={
            "help": "Do not use gradient accumulation steps for embedding."
            " Using both `no_gen_gas` and `no_emb_gas` will activate the GradCache Trainer"
            " but without gradient accumulation. This is useful for saving memory as it"
            " computes the backward for embedding first and then the backward for generative"
            " which is more memory efficient than the default of computing both losses,"
            " adding them and then doing the backward (https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch)."
        }
    )
    loss_gen_factor: float = field(default=1.0, metadata={"help": "Factor to scale generative loss by"})
    loss_gen_type: str = field(default="mixed", metadata={"help": "Type of gen loss: mixed/token"})
    lora: bool = field(default=False, metadata={"help": "Use LoRA PEFT"})
    qlora: bool = field(default=False, metadata={"help": "Use QLoRA PEFT"})
    save_safetensors: bool = field(default=False, metadata={"help": "Save in safetensors format"})
    split_emb: bool = field(default=False, metadata={"help": "Split embedding forward / backward pass"})
    split_emb_full: bool = field(default=False, metadata={"help": "Split embedding forward / backward pass"})
    emb_q_only: bool = field(default=False, metadata={"help": "Only backprop on q's"})
    emb_p_only: bool = field(default=False, metadata={"help": "Only backprop on p's (pos & neg)"})
