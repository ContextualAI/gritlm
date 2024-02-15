#!/bin/bash
CKPT=$1
# Make out be the last string after /
OUT=results/${CKPT##*/}
mkdir -p $OUT

DATADIR=/data/niklas/gritlm/evaldata/eval
FORMAT=eval.templates.create_prompt_with_gritlm_chat_format
#FORMAT=eval.templates.create_prompt_with_gritlm_zephyr_chat_format
#FORMAT=eval.templates.create_prompt_with_mistral_chat_format
#FORMAT=eval.templates.create_prompt_with_halo_chat_format
# Set to 8 for Mixtral
TP=1

pwd=$(pwd)
cd open-instruct

python -m eval.gsm.run_eval \
--data_dir $DATADIR/gsm \
--max_num_examples 200 \
--save_dir $OUT \
--model $CKPT \
--tokenizer_name_or_path $CKPT \
--n_shot 8 \
--use_vllm \
--use_chat_format \
--chat_formatting_function $FORMAT \
--tensor_parallel_size $TP

python -m eval.mmlu.run_eval \
--ntrain 0 \
--data_dir $DATADIR/mmlu \
--save_dir $OUT \
--model_name_or_path $CKPT \
--tokenizer_name_or_path $CKPT \
--eval_batch_size 4 \
--use_chat_format \
--chat_formatting_function $FORMAT

python -m eval.bbh.run_eval \
--data_dir $DATADIR/bbh \
--save_dir $OUT \
--model $CKPT \
--tokenizer_name_or_path $CKPT \
--max_num_examples_per_task 40 \
--use_vllm \
--use_chat_format \
--chat_formatting_function $FORMAT \
--tensor_parallel_size $TP

python -m eval.tydiqa.run_eval \
--data_dir $DATADIR/tydiqa \
--n_shot 1 \
--max_num_examples_per_lang 100 \
--max_context_length 512 \
--save_dir $OUT \
--model $CKPT \
--tokenizer_name_or_path $CKPT \
--use_vllm \
--use_chat_format \
--chat_formatting_function $FORMAT \
--tensor_parallel_size $TP

cd /home/niklas/gritlm/evaluation/bigcode-evaluation-harness

# If TP=8, use the below

if [ $TP -eq 8 ]; then
    python main.py \
    --model $CKPT \
    --max_length_generation 2048 \
    --tasks humanevalsynthesize-python \
    --temperature 0.2 \
    --n_samples 20 \
    --batch_size 1 \
    --allow_code_execution \
    --precision bf16 \
    --max_memory_per_gpu 'auto' \
    --save_generations_path $OUT/generations_humanevalsynthesizepython.json \
    --metric_output_path $OUT/evaluation_humanevalsynthesizepython.json \
    --save_generations \
    --prompt tulu
else
    accelerate launch --config_file /home/niklas/gritlm/scripts/configs/confnew/config_1gpusfsdp_m7.yml main.py \
    --model $CKPT \
    --tasks humanevalsynthesize-python \
    --do_sample True \
    --temperature 0.2 \
    --n_samples 20 \
    --batch_size 5 \
    --allow_code_execution \
    --save_generations \
    --trust_remote_code \
    --prompt tulu \
    --save_generations_path $OUT/generations_humanevalsynthesizepython.json \
    --metric_output_path $OUT/evaluation_humanevalsynthesizepython.json \
    --max_length_generation 2048 \
    --precision bf16
fi

: '
export OPENAI_API_KEY=YOUR_KEY
python -m eval.alpaca_farm.run_eval \
--use_vllm \
--model_name_or_path $CKPT \
--tokenizer_name_or_path $CKPT \
--save_dir $OUT \
--use_chat_format \
--chat_formatting_function eval.templates.create_prompt_with_gritlm_chat_format

python -m eval.alpaca_farm.run_eval \
--use_vllm \
--model_name_or_path $CKPT \
--tokenizer_name_or_path $CKPT \
--save_dir $OUT \
--use_chat_format \
--chat_formatting_function eval.templates.create_prompt_with_gritlm_chat_format \
--alpaca2
'
cd $pwd