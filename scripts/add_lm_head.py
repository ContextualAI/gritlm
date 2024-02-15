import torch

### Add GPT-J head to SGPT BE 5.8B ###
#MODEL_A = "/data/niklas/gritlm/SGPT-5.8B-weightedmean-msmarco-specb-bitfit/pytorch_model.bin"
#MODEL_B = "/data/niklas/gritlm/gptjtmp/pytorch_model.bin"
#MODEL_OUT = "/data/niklas/gritlm/SGPT-5.8B-weightedmean-msmarco-specb-bitfit-head/pytorch_model.bin"

### Add Mistral head to Embedding-only model ###
MODEL_A = "/data/huggingface/hub/models--ContextualAI--emb_m7_sq2048_medi_bb/snapshots/aecaedcecf69e7b94b310bcddaf3fa78e44123c6/pytorch_model.bin"
MODEL_B = "/data/niklas/gritlm/mistraltmp/pytorch_model-00002-of-00002.bin"
MODEL_OUT = "/data/niklas/gritlm/emb_m7_sq2048_medi_bb_head/pytorch_model.bin"

a = torch.load(MODEL_A)
b = torch.load(MODEL_B)
# Add lm head to a
#a = {"transformer." + k: v for k, v in a.items()}
a["lm_head.weight"] = b["lm_head.weight"]
#a["lm_head.bias"] = b["lm_head.bias"]

torch.save(a, MODEL_OUT)