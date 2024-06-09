#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import functools
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Union

import accelerate
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, DistributedType, set_seed
from huggingface_hub import create_repo
from packaging import version
from PIL import Image
from torch.utils.data import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, 
    PretrainedConfig,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    # StableDiffusionXLInpaintPipeline,
    # UNet2DConditionModel,
)
from diffusers.models import ImageProjection
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

import time

MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)


def load_embedding_model(args):
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
    )

    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, image_encoder


def log_validation(vae, unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two,
                   noise_scheduler, image_encoder, unet_encoder, sample,
                   args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    if args.use_cache_embedding:
        tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, image_encoder = load_embedding_model(args)

    unet = accelerator.unwrap_model(unet)
    pipeline = TryonPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipeline.unet_encoder = unet_encoder
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            img_emb_list = []
            for i in range(sample['cloth'].shape[0]):
                img_emb_list.append(sample['cloth'][i])
            
            prompt = sample["caption"]

            num_prompts = sample['cloth'].shape[0]                                        
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts

            image_embeds = torch.cat(img_emb_list,dim=0)

            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipeline.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
            
            
                prompt = sample["caption_cloth"]
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                if not isinstance(prompt, List):
                    prompt = [prompt] * num_prompts
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * num_prompts


                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipeline.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                    images = [[] for _ in range(len(prompt_embeds))]
                    for i in range(4):
                        images_batch = pipeline(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            num_inference_steps=args.num_inference_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=sample['pose_img'],
                            text_embeds_cloth=prompt_embeds_c,
                            cloth=sample["cloth_pure"].to(accelerator.device),
                            mask_image=sample['inpaint_mask'],
                            image=(sample['image']+1.0)/2.0, 
                            height=args.height,
                            width=args.width,
                            guidance_scale=args.guidance_scale,
                            ip_adapter_image=image_embeds,
                        )[0]
                        for j in range(len(prompt_embeds)):
                            images[j].append(images_batch[j])

    model_image = (
        ((sample['image']+1.0)/2.0 * 255.)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    masked_image = model_image * (sample['inpaint_mask'].permute(0, 2, 3,1).numpy() < 0.5)

    cloth_image = (
        ((sample['cloth_pure']+1.0)/2.0 * 255.)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    image_logs = []
    for i in range(len(images)):
        image_logs.append({
            "validation_image_model": Image.fromarray(model_image[i]).convert("RGB"), 
            "validation_image_masked": Image.fromarray(masked_image[i]).convert("RGB"), 
            "validation_image_cloth": Image.fromarray(cloth_image[i]).convert("RGB"), 
            "validation_prompt": f"batch_{i}: " + prompt[i],
            "images": images[i], 
        })

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                
                validation_prompt = log["validation_prompt"]
                validation_image_model = np.asarray(log["validation_image_model"])
                validation_image_masked = np.asarray(log["validation_image_masked"])
                validation_image_cloth = np.asarray(log["validation_image_cloth"])
                images = np.asarray(log["images"])

                formatted_images = []

                formatted_images.append(np.asarray(validation_image_model))
                formatted_images.append(np.asarray(validation_image_masked))
                formatted_images.append(np.asarray(validation_image_cloth))

                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")

        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image_model = log["validation_image_model"]
                validation_image_masked = log["validation_image_masked"]
                validation_image_cloth = log["validation_image_cloth"]

                formatted_images.append(wandb.Image(validation_image_model, caption="adapter conditioning model"))
                formatted_images.append(wandb.Image(validation_image_masked, caption="adapter conditioning model"))
                formatted_images.append(wandb.Image(validation_image_cloth, caption="adapter conditioning cloth"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(
    repo_id: str,
    images=None,
    validation_prompt=None,
    base_model=str,
    dataset_name=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
--- license: creativeml-openrail-m base_model: {base_model} dataset: {dataset_name} tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
inference: true ---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{args.dataset_name}** dataset. Below are some example images
generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n {img_str}

Special VAE used for training: {vae_path}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-inpainting",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_prodigy_optim", action="store_true", help="Whether or not to use Prodigy optimizer.")
    parser.add_argument(
        "--use_cosine_annealing_schedule", action="store_true", help="Whether or not to use cosine annealing schedule."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading."),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_batch_size", type=int, default=2, help="Batch size (per device) for the validation dataloader."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help=(
            "A set of paths to the mask conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sdxl-inpainting",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--use_euler",
        action="store_true",
        default=False,
        help="Whether or not to use Euler Scheduler.",
    )
    parser.add_argument(
        "--use_non_uniform_timesteps",
        action="store_true",
        default=False,
        help="Whether or not to use non-uniform timesteps.",
    )
    parser.add_argument("--category",type=str,default="upper_body",choices=["upper_body", "lower_body", "dresses"])
    parser.add_argument("--unpaired",action="store_true",)
    parser.add_argument("--num_inference_steps",type=int,default=30,)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--use_cache_embedding",action="store_true",)
    parser.add_argument("--cache_embedding_dir",type=str)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(
            "`--height/width` must be divisible by 8 for consistently sized encoded images between the VAE and the unet."
        )

    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id

    # Load scheduler and models
    if args.use_euler:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if not args.use_cache_embedding:
            tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, image_encoder = load_embedding_model(args)
        else:
            tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, image_encoder = None, None, None, None, None, None

        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet_encoder",
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_unet_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    # with torch.no_grad():
    #     # Increase the number of input channels in the unet to handle
    #     # the additional mask and masked image conditioning
    #     orig_in_channels = unet.config.in_channels
    #     unet.config.in_channels = 2 * orig_in_channels + 1  # 2 images + 1 mask
    #     unet.register_to_config(in_channels=unet.config.in_channels)
    #     original_conv_in = unet.conv_in
    #     unet.conv_in = torch.nn.Conv2d(
    #         unet.config.in_channels, unet.config.block_out_channels[0], kernel_size=3, padding=(1, 1)
    #     )
    #     unet.conv_in.bias = original_conv_in.bias
    #     # set first `origin_n_channels` input channels of `unet.conv_in.weight` to `original_conv_in.weight`
    #     # 2d conv weight shape: `out channels, in channels, kernel height, kernel width`
    #     unet.conv_in.weight[:, :orig_in_channels, :, :] = original_conv_in.weight
    #     unet.conv_in.weight[:, orig_in_channels:, :, :] = 0
    #     del original_conv_in

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config, inv_gamma=1, power=3 / 4
        )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    logger.info({"class_name": model.__class__.__name__})
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    unet.train()
    if not args.use_cache_embedding:
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    elif args.use_prodigy_optim:
        try:
            from prodigyopt import Prodigy
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`.")
        optimizer_class = Prodigy
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not args.use_cache_embedding:
        # Move vae, unet and text_encoder to device and cast to weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        if args.pretrained_vae_model_name_or_path is not None:
            vae.to(accelerator.device, dtype=weight_dtype)
        else:
            vae.to(accelerator.device, dtype=torch.float32)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_encoder.to(accelerator.device, dtype=weight_dtype)
    

    logger.info(f"Trainable/All unet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e06} M / {sum(p.numel() for p in unet.parameters()) / 1e06} M")
    if not args.use_cache_embedding:
        logger.info(f"Trainable/All vae: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e06} M / {sum(p.numel() for p in text_encoder_one.parameters()) / 1e06} M")
        logger.info(f"Trainable/All text_encoder_one: {sum(p.numel() for p in text_encoder_one.parameters() if p.requires_grad) / 1e06} M / {sum(p.numel() for p in text_encoder_one.parameters()) / 1e06} M")
        logger.info(f"Trainable/All text_encoder_two: {sum(p.numel() for p in text_encoder_two.parameters() if p.requires_grad) / 1e06} M / {sum(p.numel() for p in text_encoder_two.parameters()) / 1e06} M")
        logger.info(f"Trainable/All image_encoder: {sum(p.numel() for p in image_encoder.parameters() if p.requires_grad) / 1e06} M / {sum(p.numel() for p in image_encoder.parameters()) / 1e06} M")
    logger.info(f"Trainable/All unet_encoder: {sum(p.numel() for p in unet_encoder.parameters() if p.requires_grad) / 1e06} M / {sum(p.numel() for p in unet_encoder.parameters()) / 1e06} M")

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        prompt_batch, prompt_batch_cloth, original_sizes, crop_coords, proportion_empty_prompts, text_encoders, tokenizers, is_train=True
    ):
        target_size = (args.height, args.width)
        original_sizes = list(map(list, zip(*original_sizes)))
        crops_coords_top_left = list(map(list, zip(*crop_coords)))

        original_sizes = torch.tensor(original_sizes, dtype=torch.long)
        crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch.long)

        # crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds
        prompt_embeds_cloth, pooled_prompt_embeds = encode_prompt(
            prompt_batch_cloth, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        # add_time_ids = list(crops_coords_top_left + target_size)
        add_time_ids = list(target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        # add_time_ids = torch.cat([torch.tensor(original_sizes, dtype=torch.long), add_time_ids], dim=-1)
        add_time_ids = torch.cat([original_sizes, crops_coords_top_left, add_time_ids], dim=-1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, "prompt_embeds_cloth": prompt_embeds_cloth, **unet_added_cond_kwargs}

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    from inference_dc import DresscodeTestDataset
    dataset = DresscodeTestDataset(
        dataroot_path=args.train_shards_path_or_url,
        phase="train",
        category = args.category,
        size=(args.height, args.width),
        cache_embedding_dir=args.cache_embedding_dir,
        use_cache_embedding=args.use_cache_embedding
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    if accelerator.is_main_process:
        validation_dataset = DresscodeTestDataset(
            dataroot_path=args.train_shards_path_or_url,
            phase="test",
            order="unpaired" if args.unpaired else "paired",
            category = args.category,
            size=(args.height, args.width),
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=False,
            batch_size=args.validation_batch_size,
            num_workers=args.dataloader_num_workers,
        )
        sample = next(iter(validation_dataloader))

    if not args.use_cache_embedding:
        # Let's first compute all the embeddings so that we can free up the text encoders
        # from memory.
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]

        compute_embeddings_fn = functools.partial(
            compute_embeddings,
            proportion_empty_prompts=args.proportion_empty_prompts,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.use_cosine_annealing_schedule:  # to be used with Prodigy
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(unet, optimizer, lr_scheduler, train_dataloader)
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
        # data_iter = iter(train_dataloader)
        # for step in range(len(train_dataloader)):
        #     # s = time.time()
        #     batch = next(data_iter)
        #     # logger.info(f"data iterated: {time.time() - s}")
            with accelerator.accumulate(unet):
                
                image = batch['image'].to(accelerator.device, non_blocking=True) # [-1, 1]
                pose_image = batch['pose_img'].to(accelerator.device, non_blocking=True) # densepose # [-1, 1]
                cloth_image = batch["cloth_pure"].to(accelerator.device, non_blocking=True) # cloth image
                cloth_clip_image = batch['cloth'][:, 0].to(accelerator.device, non_blocking=True) # cloth image after clip processor

                mask = batch['inpaint_mask'].to(accelerator.device, non_blocking=True) # mask
                masked_image = image * (mask < 0.5)

                prompt_model = batch["caption"]
                prompt_cloth = batch["caption_cloth"]

                bsz = image.shape[0]
                if not args.use_cache_embedding:

                    # s = time.time()
                    added_cond_kwargs = compute_embeddings_fn(prompt_model, prompt_cloth, original_sizes=[[image.shape[2]]*bsz, [image.shape[3]]*bsz], crop_coords=[[0]*bsz, [0]*bsz])

                    if args.pretrained_vae_model_name_or_path is not None:
                        pixel_values = image.to(dtype=weight_dtype)
                        masked_pixel_values = masked_image.to(dtype=weight_dtype)
                        pose_pixel_values = pose_image.to(dtype=weight_dtype)
                        cloth_pixel_values = cloth_image.to(dtype=weight_dtype)
                        if vae.dtype != weight_dtype:
                            vae.to(dtype=weight_dtype)
                    else:
                        pixel_values = image
                        masked_pixel_values = masked_image
                        pose_pixel_values = pose_image
                        cloth_pixel_values = cloth_image

                    # encode pixel values with batch size of at most 8
                    latents = []
                    for i in range(0, pixel_values.shape[0], 8):
                        latents.append(vae.encode(pixel_values[i : i + 8]).latent_dist.sample())
                    latents = torch.cat(latents, dim=0)

                    masked_latents = []
                    for i in range(0, masked_pixel_values.shape[0], 8):
                        masked_latents.append(vae.encode(masked_pixel_values[i : i + 8]).latent_dist.sample())
                    masked_latents = torch.cat(masked_latents, dim=0)

                    pose_latents = []
                    for i in range(0, pose_pixel_values.shape[0], 8):
                        pose_latents.append(vae.encode(pose_pixel_values[i : i + 8]).latent_dist.sample())
                    pose_latents = torch.cat(pose_latents, dim=0)

                    cloth_latents = []
                    for i in range(0, cloth_pixel_values.shape[0], 8):
                        cloth_latents.append(vae.encode(cloth_pixel_values[i : i + 8]).latent_dist.sample())
                    cloth_latents = torch.cat(cloth_latents, dim=0)

                    latents = latents * vae.config.scaling_factor
                    masked_latents = masked_latents * vae.config.scaling_factor
                    pose_latents = pose_latents * vae.config.scaling_factor
                    cloth_latents = cloth_latents * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        latents = latents.to(weight_dtype)
                        masked_latents = masked_latents.to(weight_dtype)
                        pose_latents = pose_latents.to(weight_dtype)
                        cloth_latents = cloth_latents.to(weight_dtype)

                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                else:
                    cache_embedding_dict = batch["cache_emebdding_dict"]

                    added_cond_kwargs = dict(
                        prompt_embeds=cache_embedding_dict["prompt_embeds"].to(accelerator.device, weight_dtype),
                        prompt_embeds_cloth=cache_embedding_dict["prompt_embeds_cloth"].to(accelerator.device, weight_dtype),
                        text_embeds=cache_embedding_dict["text_embeds"].to(accelerator.device, weight_dtype),
                        time_ids=cache_embedding_dict["time_ids"].to(accelerator.device, weight_dtype),
                    )

                    latents = cache_embedding_dict["latents"].to(accelerator.device, weight_dtype)
                    masked_latents = cache_embedding_dict["masked_latents"].to(accelerator.device, weight_dtype)
                    pose_latents = cache_embedding_dict["pose_latents"].to(accelerator.device, weight_dtype)
                    cloth_latents = cache_embedding_dict["cloth_latents"].to(accelerator.device, weight_dtype)

                    vae_scale_factor = 8

                # scale mask to match latents resolution
                latent_dimension_height = args.height // vae_scale_factor
                latent_dimension_width = args.width // vae_scale_factor
                mask = F.interpolate(mask.to(torch.float32), size=(latent_dimension_height, latent_dimension_width))
                mask = mask.to(latents.dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # Cubic sampling to sample a random timestep for each image
                # timesteps = torch.rand((bsz,), device=latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.use_euler:
                    sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                else:
                    inp_noisy_latents = noisy_latents

                model_input = torch.cat([inp_noisy_latents, mask, masked_latents, pose_latents], dim=1)

                # GarmentNet
                prompt_embeds_cloth = added_cond_kwargs.pop("prompt_embeds_cloth")
                _, reference_features = unet_encoder(cloth_latents, timesteps, prompt_embeds_cloth, return_dict=False)
                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.proportion_empty_prompts is not None:
                    random_p = torch.rand(bsz, device=latents.device)
                    reference_mask = random_p < args.proportion_empty_prompts
                    reference_mask = reference_mask.reshape(bsz, 1, 1)
                    reference_features = [reference_mask * reference_feature for reference_feature in reference_features]

                # IP-Adapter
                output_hidden_state = not isinstance(unet.module.encoder_hid_proj, ImageProjection)
                if not args.use_cache_embedding:
                    if output_hidden_state:
                        image_embeds_cloth = image_encoder(cloth_clip_image, output_hidden_states=True).hidden_states[-2]
                    else:
                        image_embeds_cloth = image_encoder(cloth_clip_image).image_embeds
                else:
                    assert output_hidden_state
                    image_embeds_cloth = cache_embedding_dict["image_embeds_cloth"].to(accelerator.device, weight_dtype)

                # # project outside for loop
                # image_embeds_cloth = image_embeds_cloth.to(unet.dtype)
                # image_embeds_cloth = unet.module.encoder_hid_proj(image_embeds_cloth).to(prompt_embeds_cloth.dtype)
                added_cond_kwargs["image_embeds"] = image_embeds_cloth

                # logger.info(f"latent prepared: {time.time() - s}")
                # s = time.time()

                # predict the noise residual
                prompt_embeds = added_cond_kwargs.pop("prompt_embeds")
                model_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    garment_features=reference_features,
                ).sample

                # logger.info(f"unet forwarded: {time.time() - s}")
                # s = time.time()

                if args.use_euler:
                    model_pred = model_pred * (-sigmas) + noisy_latents
                    weighing = sigmas**-2.0

                # Get the target for loss depending on the prediction type
                target = latents if args.use_euler else noise

                if args.use_euler:
                    loss = torch.mean(
                        (weighing.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
                    )
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                # logger.info(f"loss backwarded: {time.time() - s}")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0 or global_step == 1:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        if args.use_ema:
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        log_validation(vae, unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two,
                                       noise_scheduler, image_encoder, unet_encoder, sample,
                                       args, accelerator, weight_dtype, global_step)

                        # Switch back to the original UNet parameters.
                        if args.use_ema:
                            ema_unet.restore(unet.parameters())

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
            unet.save_pretrained(os.path.join(args.output_dir, "unet_ema"))

        # if args.push_to_hub:
        #     save_model_card(
        #         repo_id,
        #         image_logs=image_logs,
        #         base_model=args.pretrained_model_name_or_path,
        #         repo_folder=args.output_dir,
        #     )
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)