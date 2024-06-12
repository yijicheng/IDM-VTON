# export MODEL_NAME="yisol/IDM-VTON"
# export UNET_NAME="yisol/IDM-VTON-DC"

export MODEL_NAME="/root/.cache/huggingface/hub/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/"
export UNET_NAME="/root/.cache/huggingface/hub/models--yisol--IDM-VTON-DC/snapshots/0fcf915a04a97a353678e2f17f89587127fce7f0/"
# export UNET_NAME="/root/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16/checkpoint-3000"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_PATH="../../data/ft_local/DressCode"

/root/miniconda3/envs/idm/bin/accelerate launch  \
    --main_process_port=29500 \
    --config_file="training_configs/accelerate_config_ds.yaml" \
    train.py  \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_unet_model_name_or_path=$UNET_NAME \
    --train_shards_path_or_url=$DATASET_PATH \
    --width 768 --height 1024 \
    --proportion_empty_prompts=0.2 \
    --dataloader_num_workers=8 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=100000 \
    --learning_rate=1e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --seed=42 \
    --report_to="tensorboard" \
    --checkpointing_steps=1000 --checkpoints_total_limit=100 \
    --validation_steps=200 \
    --output_dir="outputs/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16-add_noise_t_to_densepose_latent" \
    --logging_dir="logs" \
    --tracker_project_name="sdxl-inpainting" \
    --category="lower_body" --unpaired \
    --cache_embedding_dir="../../data/DressCode-cache-embedding" --use_cache_embedding \
    --densepose_aug_type="add_noise_t_to_densepose_latent" \
    --resume_from_checkpoint=latest


#   --enable_xformers_memory_efficient_attention \
#   --use_8bit_adam \
#   --pretrained_vae_model_name_or_path=$VAE_NAME \
