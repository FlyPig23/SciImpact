#!/bin/bash

export WANDB_PROJECT="paper_impact_sft"

export WANDB_API_KEY=$(cat ../.wandb_api_key)

if [ -f ../.hf_token ]; then
    export HF_TOKEN=$(cat ../.hf_token)
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES=0,1,6,7

export DISABLE_VERSION_CHECK=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset paper_impact_sft_train \
    --dataset_dir data \
    --eval_dataset paper_impact_sft_val \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/Qwen3-4B_Paper_Impact_SFT \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --flash_attn sdpa \
    --deepspeed ds_zero2.json \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 500 \
    --report_to wandb \
    --run_name Qwen3-4B_Paper_Impact_SFT \
    --resume_from_checkpoint saves/Qwen3-4B_Paper_Impact_SFT/checkpoint-4000 > training_resume_2.log 2>&1 &
