"""
Compute median AlpacaEval generation length. Simply change the path and run:
`python alpaca_median.py`
"""
import json

path = "results/gritlm_m7_sq2048_e5_bbcc_ds_gfix/alpaca_eval-gritlm_m7_sq2048_e5_bbcc_ds_gfix-greedy-long-output.json"
path = "results/gritlm_m7_sq2048_e5_bbcc_ds_bs2048/alpaca_eval-gritlm_m7_sq2048_e5_bbcc_ds_bs2048-greedy-long-output.json"
path = "results/gritlm_m7_sq2048_medi2_bbcc_bs4096/alpaca_eval-gritlm_m7_sq2048_medi2_bbcc_bs4096-greedy-long-output.json"
path = "results/gritlm_m7_sq512_medi2_bbcc/alpaca_eval-gritlm_m7_sq512_medi2_bbcc-greedy-long-output.json"

with open(path, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

# Median length
lengths = [len(d["output"]) for d in data]
print("Median length:", sorted(lengths)[len(lengths) // 2])