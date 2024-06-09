export MODEL_NAME="yisol/IDM-VTON"
export UNET_NAME="yisol/IDM-VTON-DC"
# export MODEL_NAME="/root/.cache/huggingface/hub/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/"
# export UNET_NAME="/root/.cache/huggingface/hub/models--yisol--IDM-VTON-DC/snapshots/0fcf915a04a97a353678e2f17f89587127fce7f0/"
export DATASET_PATH="../../data/ft_local/DressCode"


# /root/miniconda3/envs/idm/bin/accelerate launch \
#     --main_process_port=29501 \
#     inference_dc.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --pretrained_unet_model_name_or_path=$UNET_NAME \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "results_paired_lower" \
#     --data_dir ${DATASET_PATH} \
#     --seed 42 \
#     --test_batch_size 4 \
#     --guidance_scale 2.0 \
#     --category "lower_body" 

# /root/miniconda3/envs/idm/bin/accelerate launch \
#     --main_process_port=29501 \
#     inference_dc.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --pretrained_unet_model_name_or_path=$UNET_NAME \
#     --width 768 --height 1024 --num_inference_steps 30 \
#     --output_dir "results_unpaired_lower" \
#     --unpaired \
#     --data_dir ${DATASET_PATH} \
#     --seed 42 \
#     --test_batch_size 2 \
#     --guidance_scale 2.0 \
#     --category "lower_body" 

/root/miniconda3/envs/idm/bin/accelerate launch \
    --main_process_port=29501 \
    inference_dc_cascade.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_unet_model_name_or_path=$UNET_NAME \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "results_cascade" \
    --data_dir ${DATASET_PATH} \
    --seed 42 \
    --test_batch_size 4 \
    --guidance_scale 2.0

