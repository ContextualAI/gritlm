"""
Sources
- https://discuss.huggingface.co/t/how-to-load-a-checkpoint-model-with-sharded-state-dict/62448
- https://github.com/facebookresearch/llama-recipes/blob/main/docs/inference.md#loading-back-fsdp-checkpoints
- https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py
"""
import fire

import torch
import torch.distributed.checkpoint as dist_cp

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel


class AutoModelForCausalLMWrapper(torch.nn.Module):
    TRANSFORMER_CLS = AutoModel
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_config(*args, **kwargs)
        

def load_sharded_model_single_gpu(model, model_path):
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(model_path),
        no_dist=True,
    )
    
    result = model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    print(result)
    return model

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def convert_checkpoint(hf_model: str, fsdp_model_path: str, output_path: str, pipeline_parallel=False):
    '''
    hf_model: transformers path.
    fsdp_model_path: path to the fsdp checkpoint, for example `/x/checkpoint-xxx/pytorch_model_x`
    output_path: output path to save the converted checkpoint
    '''
    config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

    kwargs = {'trust_remote_code': True}
    if pipeline_parallel:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = get_gpus_max_memory("50GB")
        kwargs["offload_folder"] = "offload"

    model = AutoModelForCausalLMWrapper(config, **kwargs)
    #model = AutoModelForCausalLM.from_config(config, **kwargs)

    model = load_sharded_model_single_gpu(model, fsdp_model_path)
    #model.save_pretrained(output_path, max_shard_size="10GB")
    # Reformat
    model.model.save_pretrained(output_path, max_shard_size="10GB")
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    fire.Fire(convert_checkpoint)