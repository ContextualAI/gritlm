#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 99:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive
#SBATCH --array=0-49%50

######################
### Set enviroment ###
######################
cd /home/niklas/gritlm
source /env/bin/start-ctx-user
conda activate gritlm
export WANDB_PROJECT="gritlm"
######################

PARAMS=(
1,250,query
1,500,query
1,1000,query
1,2000,query
1,4000,query
250,1,query
500,1,query
1000,1,query
2000,1,query
4000,1,query
1,250,doc
1,500,doc
1,1000,doc
1,2000,doc
1,4000,doc
250,1,doc
500,1,doc
1000,1,doc
2000,1,doc
4000,1,doc
1,250,querydoc
1,500,querydoc
1,1000,querydoc
1,2000,querydoc
1,4000,querydoc
250,1,querydoc
500,1,querydoc
1000,1,querydoc
2000,1,querydoc
4000,1,querydoc
1,250,docquery
1,500,docquery
1,1000,docquery
1,2000,docquery
1,4000,docquery
250,1,docquery
500,1,docquery
1000,1,docquery
2000,1,docquery
4000,1,docquery
1,250,"None"
1,500,"None"
1,1000,"None"
1,2000,"None"
1,4000,"None"
250,1,"None"
500,1,"None"
1000,1,"None"
2000,1,"None"
4000,1,"None"
)

CONF=${PARAMS[$SLURM_ARRAY_TASK_ID]}
echo $CONF
IFS=',' read q d c <<< "${CONF}"

# For no retrieval add `--no_retrieval` to both commands
# For doc then query prompt add `--prompt doc` to both commands
# GPU
python rag/eval.py \
--model_name_or_path /data/niklas/gritlm/gritlm_m7_sq2048_e5ds_bbcc_bs2048_gc \
--eval_data /data/niklas/gritlm/rag/nq_data/test.jsonl \
--passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl \
--customd $d \
--customq $q \
--limit 1 \
--latency \
--min_new_tokens 16 \
--max_new_tokens 16 \
--prompt doc

# CPU
CUDA_VISIBLE_DEVICES="" python rag/eval.py \
--model_name_or_path /data/niklas/gritlm/gritlm_m7_sq2048_e5ds_bbcc_bs2048_gc \
--eval_data /data/niklas/gritlm/rag/nq_data/test.jsonl \
--passages /data/niklas/gritlm/rag/corpora/nqbeir/corpus.jsonl \
--customd $d \
--customq $q \
--limit 1 \
--latency \
--min_new_tokens 16 \
--max_new_tokens 16 \
--prompt doc
