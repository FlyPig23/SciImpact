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
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset paper_impact_sft_train \
    --dataset_dir data \
    --eval_dataset paper_impact_sft_val \
    --template llama3 \
    --finetuning_type full \
    --output_dir saves/Llama3.2-3B_Paper_Impact_SFT_hold \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
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
    --run_name Llama3.2-3B_Paper_Impact_SFT_hold > train_log_llama_hold.log 2>&1 &
    # --resume_from_checkpoint saves/Llama3.2-3B_Paper_Impact_SFT/checkpoint-3000 > llama_training_resume.log 2>&1 &
