#!/bin/bash

JSON_FOLDER="askchart-data/train_chart_json"
IMAGE_FOLDER="askchart-data"
cd /hpc2hdd/home/askchart/sootung/AskChart
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed askchart/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /hpc2hdd/home/askchart/sootung/models/LanguageBind/MoE-LLaVA-Phi2-Stage2-384 \
    --version plain \
    --data_path ${JSON_FOLDER}/instruct_chart2table_OCR.json \
    --tune_image_tower True \
    --tune_entire_model False \
    --tune_vit_from_layer -1 \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower 'google/siglip-so400m-patch14-384' \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/llavaphi-2.7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

python scripts/v2/change_image_tower.py --path /hpc2hdd/home/askchart/sootung/models/LanguageBind/MoE-LLaVA-Phi2-Stage2-384

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed askchart/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /hpc2hdd/home/askchart/sootung/models/LanguageBind/MoE-LLaVA-Phi2-Stage2-384 \
    --version phi \
    --data_path ${JSON_FOLDER}/instruct_opencqa_OCR.json ${JSON_FOLDER}/instruct_single_round_OCR.json ${JSON_FOLDER}/instruct_chartqa_OCR.json \
                ${JSON_FOLDER}/chartqa_vp_reconstruct_shuffle_post.json ${JSON_FOLDER}/instruct_img_info_post_part1_OCR.json ${JSON_FOLDER}/instruct_img_info_post_part2_OCR.json \
    --tune_image_tower True \
    --tune_entire_model True \
    --tune_vit_from_layer -1 \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ./checkpoints/llavaphi-2.7b-pretrain/image_tower \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/llavaphi-2.7b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaphi-2.7b-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 25000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

python scripts/v2/change_image_tower.py --path checkpoints/llavaphi-2.7b-finetune --mm_image_tower './checkpoints/llavaphi-2.7b-finetune/image_tower'

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed askchart/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules fc1 fc2 wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llavaphi-2.7b-finetune \
    --version phi \
    --data_path ${JSON_FOLDER}/instruct_ChartQA_train_OCR.json ${JSON_FOLDER}/instruct_chartQA_2table_train_augmented_OCR.json ${JSON_FOLDER}/instruct_chartQA_2table_train_human_OCR.json \
                ${JSON_FOLDER}/instruct_spider_2table_cot.json ${JSON_FOLDER}/instruct_bird_train_2table_cot.json ${JSON_FOLDER}/instruct_bird_dev_2table_cot.json \
                ${JSON_FOLDER}/instruct_chart2text_dataset_OCR.json ${JSON_FOLDER}/instruct_OpenCQA_train_data.json \
    --tune_image_tower True \
    --tune_entire_model False \
    --tune_vit_from_layer -1 \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ./checkpoints/llavaphi-2.7b-finetune/image_tower \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaphi-2.7b-finetune-moe \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
