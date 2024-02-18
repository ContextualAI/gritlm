#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 99:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive
#SBATCH --array=0-68%69

# FEWSHOT: 0-9%10

######################
### Set enviroment ###
######################
cd /home/niklas/gritlm
source /env/bin/start-ctx-user
conda activate gritlmt2
export WANDB_PROJECT="gritlm"
######################

######################
#### Set network #####
######################

######################

ALLDS=(
AmazonCounterfactualClassification
AmazonPolarityClassification
AmazonReviewsClassification
ArguAna
ArxivClusteringP2P
ArxivClusteringS2S
AskUbuntuDupQuestions
BIOSSES
Banking77Classification
BiorxivClusteringP2P
BiorxivClusteringS2S
CQADupstackAndroidRetrieval
CQADupstackEnglishRetrieval
CQADupstackGamingRetrieval
CQADupstackGisRetrieval
CQADupstackMathematicaRetrieval
CQADupstackPhysicsRetrieval
CQADupstackProgrammersRetrieval
CQADupstackStatsRetrieval
CQADupstackTexRetrieval
CQADupstackUnixRetrieval
CQADupstackWebmastersRetrieval
CQADupstackWordpressRetrieval
ClimateFEVER
DBPedia
EmotionClassification
FEVER
FiQA2018
HotpotQA
ImdbClassification
MSMARCO
MTOPDomainClassification
MTOPIntentClassification
MassiveIntentClassification
MassiveScenarioClassification
MedrxivClusteringP2P
MedrxivClusteringS2S
MindSmallReranking
NFCorpus
NQ
QuoraRetrieval
RedditClustering
RedditClusteringP2P
SCIDOCS
SICK-R
STS12
STS13
STS14
STS15
STS16
STS17
STS22
STSBenchmark
SciDocsRR
SciFact
SprintDuplicateQuestions
StackExchangeClustering
StackExchangeClusteringP2P
StackOverflowDupQuestions
SummEval
TRECCOVID
Touche2020
ToxicConversationsClassification
TweetSentimentExtractionClassification
TwentyNewsgroupsClustering
TwitterSemEval2015
TwitterURLCorpus
)

# Dataets for fewshot exps
FEWSHOT=(
Banking77Classification
EmotionClassification
ImdbClassification
BiorxivClusteringS2S
SprintDuplicateQuestions
TwitterSemEval2015
TwitterURLCorpus
AskUbuntuDupQuestions
ArguAna
SCIDOCS
STS12
SummEval
)

DS=${ALLDS[$SLURM_ARRAY_TASK_ID]}

echo "Running $DS"

### M8X7 ###
python evaluation/eval_mteb.py \
--model_name_or_path /data/niklas/gritlm/m8x7_nodes32_400_fast \
--instruction_set e5 \
--instruction_format gritlm \
--task_names $DS \
--batch_size 64 \
--pipeline_parallel \
--attn_implementation sdpa \
--pooling_method mean

python evaluation/eval_mteb.py \
--model_name_or_path /data/niklas/gritlm/m8x7_nodes32_400_fast \
--instruction_set e5 \
--instruction_format gritlm \
--task_names $DS \
--batch_size 32 \
--pipeline_parallel \
--attn_implementation sdpa \
--pooling_method mean

python evaluation/eval_mteb.py \
--model_name_or_path /data/niklas/gritlm/m8x7_nodes32_400_fast \
--instruction_set e5 \
--instruction_format gritlm \
--task_names $DS \
--batch_size 16 \
--pipeline_parallel \
--attn_implementation sdpa \
--pooling_method mean

python evaluation/eval_mteb.py \
--model_name_or_path /data/niklas/gritlm/m8x7_nodes32_400_fast \
--instruction_set e5 \
--instruction_format gritlm \
--task_names $DS \
--batch_size 8 \
--pipeline_parallel \
--attn_implementation sdpa \
--pooling_method mean

python evaluation/eval_mteb.py \
--model_name_or_path /data/niklas/gritlm/m8x7_nodes32_400_fast \
--instruction_set e5 \
--instruction_format gritlm \
--task_names $DS \
--batch_size 4 \
--pipeline_parallel \
--attn_implementation sdpa \
--pooling_method mean


python evaluation/eval_mteb.py \
--model_name_or_path /data/niklas/gritlm/m8x7_nodes32_400_fast \
--instruction_set e5 \
--instruction_format gritlm \
--task_names $DS \
--batch_size 2 \
--pipeline_parallel \
--attn_implementation sdpa \
--pooling_method mean

### FP32 ###
#python /home/niklas/gritlm/evaluation/eval_mteb.py \
#--model_name_or_path /data/niklas/gritlm/gritlm_m7_sq2048_e5ds_bbcc_bs2048_token_nodes16_gen02_fp32 \
#--task_names $DS \
#--batch_size 8 \
#--pooling_method mean \
#--instruction_set e5 \
#--instruction_format gritlm \
#--attn bbcc \
#--attn_implementation sdpa \
#--dtype float32

### E5S BF16 ### 
# For 1-shot add `--num_shots 1 \`
#python /home/niklas/gritlm/evaluation/eval_mteb.py \
#--model_name_or_path /data/niklas/gritlm/m7_nodes8_rerun \
#--task_names $DS \
#--batch_size 32 \
#--pooling_method mean \
#--instruction_set e5 \
#--instruction_format gritlm \
#--attn bbcc \
#--attn_implementation sdpa \
#--dtype bfloat16

### MEDI2 ###
# For 1-shot add `--num_shots 1 \`
#python /home/niklas/gritlm/evaluation/eval_mteb.py \
#--model_name_or_path /data/niklas/gritlm/gritlm_m7_sq2048_medi2_bbcc \
#--task_names $DS \
#--batch_size 32 \
#--pooling_method mean \
#--instruction_set medi2 \
#--instruction_format gritlm \
#--attn bbcc \
#--attn_implementation sdpa \
#--dtype bfloat16 \
#--output_folder /home/niklas/gritlm/gritlmresults/gritlm_m7_sq2048_medi2_bbcc

### Llama 70B ###
#python evaluation/eval_mteb.py \
#--model_name_or_path meta-llama/Llama-2-70b-hf \
#--no_instruction \
#--task_names $DS \
#--batch_size 4 \
#--pipeline_parallel \
#--attn_implementation sdpa \
#--pooling_method weightedmean \
#--attn cccc
