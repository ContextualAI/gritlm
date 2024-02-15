import sys
import statistics
import json
import io
import os
import csv

DS_TO_NAME = {
    "arc_challenge": "ARC-C",
    "arc_easy": "ARC-E",
    "boolq": "BoolQ",
    "piqa": "PIQA",
    "winogrande": "Winogrande",
}

results_folder = sys.argv[1]

MODELS = [
    "gritlm_mist_sq1024_multiturn_rerun",
    "gritlm_lma2_sq1024_multiturn",
    "gritlm_gptj_sq1024_multiturn",
]


scores = {}


for m in MODELS:
    results_file = os.path.join(results_folder, m, "rank_eval.json")
    with io.open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)["results"]

    scores_to_avg = []
    for ds_name, v in sorted(results.items()):
        for metric, score in sorted(v.items()):
            # LM Eval Harness Accuracy
            if metric == "acc":
                ds_name_norm = DS_TO_NAME[ds_name]
                scores.setdefault(ds_name_norm, [])
                score_norm = round(score*100, 2)
                scores[ds_name_norm].append(score_norm)
                scores_to_avg.append(score)
    scores.setdefault("Average", [])
    scores["Average"].append(round(statistics.mean(scores_to_avg)*100, 2))

print("Model & " + " & ".join(MODELS)  + " \\\\")
for ds_name, v in sorted(scores.items()):
    if ds_name == "Average": continue
    print(ds_name + " & " + " & ".join([str(x) for x in v]) + " \\\\")
print("Average & " + " & ".join([str(x) for x in scores["Average"]])  + " \\\\")