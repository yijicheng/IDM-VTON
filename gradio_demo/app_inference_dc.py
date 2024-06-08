import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


# base_path = 'yisol/IDM-VTON'
base_path = '../../model/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    # base_path,
    "../../model/models--yisol--IDM-VTON-DC/snapshots/0fcf915a04a97a353678e2f17f89587127fce7f0",
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder


# category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

def start_tryon(dict,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed, category):
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        # mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask, mask_gray = get_mask_location('dc', category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray
    # return images[0], mask_gray

# garm_list = os.listdir(os.path.join(example_path,"cloth"))
# garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]
garm_list_path = [
    "../../data/ft_local/sport-case/upper-garment-1.jpg",
    "../../data/ft_local/sport-case/upper-garment-1.jpg",
    "../../data/ft_local/sport-case/upper-garment-1.jpg",
    "../../data/ft_local/sport-case/upper-garment-1.jpg",

    "../../data/ft_local/sport-case/upper-garment-2.jpg",
    "../../data/ft_local/sport-case/upper-garment-2.jpg",
    "../../data/ft_local/sport-case/upper-garment-2.jpg",
    "../../data/ft_local/sport-case/upper-garment-2.jpg",

    "../../data/ft_local/sport-case/upper-garment-3.jpg",
    "../../data/ft_local/sport-case/upper-garment-3.jpg",
    "../../data/ft_local/sport-case/upper-garment-3.jpg",
    "../../data/ft_local/sport-case/upper-garment-3.jpg",

    "../../data/ft_local/sport-case/upper-garment-4.jpg",
    "../../data/ft_local/sport-case/upper-garment-4.jpg",
    "../../data/ft_local/sport-case/upper-garment-4.jpg",
    "../../data/ft_local/sport-case/upper-garment-4.jpg",

    "../../data/ft_local/sport-case/upper-garment-5.jpg",
    "../../data/ft_local/sport-case/upper-garment-5.jpg",
    "../../data/ft_local/sport-case/upper-garment-5.jpg",
    "../../data/ft_local/sport-case/upper-garment-5.jpg",

]
garm_ex_list = [garm_path for garm_path in garm_list_path]

# human_list = os.listdir(os.path.join(example_path,"human"))
# human_list_path = [os.path.join(example_path,"human",human) for human in human_list]
human_list_path = [
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情不笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情不笑.jpg/out_hd_0.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情笑.jpg/out_hd_1.png",

    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情不笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情不笑.jpg/out_hd_0.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情笑.jpg/out_hd_1.png",

    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情不笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情不笑.jpg/out_hd_0.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情笑.jpg/out_hd_1.png",

    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情不笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景复杂+表情笑.jpg/out_hd_1.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情不笑.jpg/out_hd_0.png",
    "../../code/IDM-VTON/outputs/lower-garment-4.jpg/全身照+背景简单+表情笑.jpg/out_hd_1.png",

]
human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

print(garm_list_path)
print(human_list_path)


##default human
for i in range(len(human_ex_list)):
    # output_dir = os.path.join(f"outputs/{garm_ex_list[i].split('/')[-1]}/{human_ex_list[i]['background'].split('/')[-1]}")

    output_dir = os.path.join(f"outputs/{garm_ex_list[i].split('/')[-1]}/{human_ex_list[i]['background'].split('/')[-3:]}")
    os.makedirs(output_dir, exist_ok=True)
    human_ex_list[i]['background'] = Image.open(human_ex_list[i]['background'])
    imgs=human_ex_list[i]
    garm_img=Image.open(garm_ex_list[i])
    prompt=""
    is_checked=True
    is_checked_crop=False
    denoise_steps=30
    seed=None

    for image_idx in range(4):
        image_out, masked_img = start_tryon(imgs, garm_img, prompt, is_checked, is_checked_crop, denoise_steps, seed, 0)
        image_out.save(os.path.join(output_dir, 'out_hd' + '_' + str(image_idx) + '.png'))

    masked_img.save(os.path.join(output_dir, "masked_img.png"))

