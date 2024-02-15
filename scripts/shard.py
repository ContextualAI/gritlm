import sys
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    sys.argv[1],
    torch_dtype="auto",
)
output_path = sys.argv[2]
model.save_pretrained(
    output_path, 
    max_shard_size="5GB",
    safe_serialization=False,
)