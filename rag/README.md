## RAG with GRIT

This document details how to reproduce the RAG experiments in the paper. The result files are also contained in the results folder: https://huggingface.co/datasets/GritLM/results/tree/main/GritLM-7B

### Setup

#### Packages

To setup packages etc. follow the instructions of the main README.md.

#### Index

You don't need the index for latency benchmarking but do need it for performance benchmarking.
We have uploaded the index of GritLM-7B here: https://huggingface.co/datasets/GritLM/index
You can also follow the below to recreate it:
```bash
python /home/niklas/gritlm/rag/prepare_qa.py --output_directory /data/niklas/gritlm/rag
wget https://huggingface.co/datasets/BeIR/nq/resolve/main/corpus.jsonl.gz
gunzip corpus.jsonl.gz
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages corpus.jsonl --save_index_path index_nq
```

### Benchmarking

To run the latency benchmark, do `bash scripts/raglatency.sh` after adjusting the script to your cluster / paths. For performance benchmarking, you can adapt & run the scripts below:

No retrieval:
```bash
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --no_retrieval --load_index_path index_nq --cache query
```

Query then document prompt RAG:
```bash
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl --load_index_path index_nq --prompt query
```

Query Caching
```bash
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl --load_index_path index_nq --cache query
```

Query-Doc Caching
```bash
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl --load_index_path index_nq --cache querydoc
```


Document then query prompt RAG:
```bash
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl --load_index_path index_nq --prompt doc
```

Doc Caching:
```bash
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl --load_index_path index_nq --cache doc
```

Doc-Query Caching
```bash
python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl --load_index_path index_nq --cache docquery
```

### Acknowledgements

The code is adapted from [ATLAS](https://github.com/facebookresearch/atlas).