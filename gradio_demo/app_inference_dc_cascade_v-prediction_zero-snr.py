import sys
sys.path.append('./')
import logging
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
import diffusers
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
import transformers
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import torchvision

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from facemasker import FaceMaskerV2, get_face_mask

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__, log_level="INFO")

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


def resize_and_paste(image, scale=0.5):
    # 打开图像
    # img = Image.open(image_path)

    # 获取原始图像的尺寸
    width, height = image.size

    # 计算缩小后的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩小图像
    resized_img = image.resize((new_width, new_height))

    # 创建一个白色背景
    background = Image.new('RGB', (width, height), (255, 255, 255))

    # 计算粘贴位置
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2

    # 把缩小后的图像粘贴到背景上
    background.paste(resized_img, (paste_x, paste_y))

    # # 保存结果
    # background.save("result.png")
    return background.convert("RGB")

    # # 使用示例
    # image_path = "path/to/your/image.jpg"
    # scale = 0.5  # 缩小倍率，例如0.5表示缩小一半
    # resize_and_paste(image_path, scale)


def main():
    
    # output_dir = "results/cascade_inference/idm-vton-dc/dresscode_upper_human_densepose"
    # output_dir = "results/cascade_inference/idm-vton-dc/test_ali"
    # output_dir = "results/cascade_inference/idm-vton-dc/debug"
    # output_dir = "results/cascade_inference/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16/checkpoint-3000/test_ali"
    # output_dir = "results/cascade_inference/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16-resume-3000/checkpoint-2000/test_ali"
    # output_dir = "results/cascade_inference/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16-resume-3000/checkpoint-2000/debug"
    output_dir = "results/cascade_inference/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16-v-prediction-zero-snr-dev/checkpoint-3000/debug"

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir)
    accelerator = Accelerator(
        project_config=accelerator_project_config,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    set_seed(42)
    device = accelerator.device

    # base_path = 'yisol/IDM-VTON'
    base_path = '/root/.cache/huggingface/hub/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/'
    example_path = os.path.join(os.path.dirname(__file__), 'example')

    unet = UNet2DConditionModel.from_pretrained(
        # base_path,
        # "/root/.cache/huggingface/hub/models--yisol--IDM-VTON-DC/snapshots/0fcf915a04a97a353678e2f17f89587127fce7f0/",
        # "/root/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16/checkpoint-3000",
        # "/root/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16-resume-3000/checkpoint-2000/",
        "/root/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16-v-prediction-zero-snr-dev/checkpoint-3000",
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

    parsing_model = Parsing(device)
    openpose_model = OpenPose(device)

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
    scheduler_args = {"prediction_type": "v_prediction", "timestep_spacing": "trailing"}
    pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config, **scheduler_args)

    # category_dict = ['upperbody', 'lowerbody', 'dress']
    category_dict_utils = ['upper_body', 'lower_body', 'dresses']

    def start_tryon(human_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed, category, neg_garment_des=""):
        
        openpose_model.preprocessor.body_estimation.model.to(device)
        pipe.to(device)
        pipe.unet_encoder.to(device)

        garm_img= garm_img.convert("RGB").resize((768,1024))
        # human_img_orig = dict["background"].convert("RGB")
        human_img_orig = human_img.convert("RGB")     
        
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
        # else:
        #     mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
        #     # mask = transforms.ToTensor()(mask)
        #     # mask = mask.unsqueeze(0)
        mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)

        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        

        args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args,human_img_arg)    
        pose_img = pose_img[:,:,::-1]    
        pose_img = Image.fromarray(pose_img).resize((768,1024))
        pose_img_pil = pose_img
        
        with torch.no_grad():
            # Extract the images
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = neg_garment_des + "monochrome, lowres, bad anatomy, worst quality, low quality"
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
                        negative_prompt = neg_garment_des + "monochrome, lowres, bad anatomy, worst quality, low quality"
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
                            guidance_rescale=0.7,
                        )[0]

        if is_checked_crop:
            out_img = images[0].resize(crop_size)        
            human_img_orig.paste(out_img, (int(left), int(top)))    
            return human_img_orig, mask, mask_gray, pose_img_pil
        else:
            return images[0], mask, mask_gray, pose_img_pil

    use_manual_path = True


    use_dresscode = False
    use_test_ali = False
    use_upper_human = True

    if use_manual_path:
        examples = [
            # (
            #     "../../data/ft_local/sport-case/upper-garment-4.jpg",
            #     "",
            #     "../../data/ft_local/sport-case/lower-garment-4.jpg",
            #     "A pair of blue shorts with the nike logo",
            #     "../../data/ft_local/照片素材-John/全身照+背景复杂+表情笑.jpg",
            # ),

            (                
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/049910_1.jpg",
                "3/4 Sleeve Square Neck Blouse",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/lower_body/images/013566_1.jpg",
                "Tapered Knee Length Jeans",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/048409_0.jpg",
            ),
            (                
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/048853_1.jpg",
                "Sleeveless Square Neck Bra Top",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/lower_body/images/013574_1.jpg",
                "Straight Knee Length Jeans",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/048417_0.jpg",
            ),
            (
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/049213_1.jpg",
                "Short Sleeves Round Neck T-Shirt",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/lower_body/images/013605_1.jpg",
                "Boots Cut Long Jeans",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/048451_0.jpg",
            ),
            (
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/049424_1.jpg",
                "Long Sleeves Shirt Collar Shirt",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/lower_body/images/013614_1.jpg",
                "Bell-Bottom / Flare Cropped Wide Leg Pants",
                "/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images/048460_0.jpg",
            )
        ]

    if use_dresscode:
        
        if use_test_ali:
            human_list = sorted(os.listdir("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/test_ali"))
        else:
            human_list = []

        upper_garment_list = []
        upper_test_pairs_unpaired = os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode" , "upper_body", f"test_pairs_unpaired.txt")
        with open(upper_test_pairs_unpaired, "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                
                if use_upper_human:
                    human_list.append(im_name)
                upper_garment_list.append(c_name)

        lower_garment_list = []
        lower_test_pairs_unpaired = os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode" , "lower_body", f"test_pairs_unpaired.txt")
        with open(lower_test_pairs_unpaired, "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()

                # human_list.append(im_name)
                lower_garment_list.append(c_name)

        lower_garment_list = sorted([x for x in os.listdir("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/lower_body/images") if x.endswith('_1.jpg')], key=lambda x: x.split("_")[0])

        upper_prompt_path = os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode" , "upper_body", "dc_caption.txt")
        lower_prompt_path = os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode" , "lower_body", "dc_caption.txt")

        upper_annotation_pair = {}
        with open(upper_prompt_path, "r") as file:
            for line in file:
                parts = line.strip().split(" ")
                upper_annotation_pair[parts[0]] = ' '.join(parts[1:])

        lower_annotation_pair = {}
        with open(lower_prompt_path, "r") as file:
            for line in file:
                parts = line.strip().split(" ")
                lower_annotation_pair[parts[0]] = ' '.join(parts[1:])

        upper_garment_list = upper_garment_list[:len(human_list)][:100]
        lower_garment_list = lower_garment_list[:len(human_list)][:100]
        human_list = human_list[:100]

        examples = []
        for upper_garment, lower_garment, human in zip(upper_garment_list, lower_garment_list, human_list):
            examples.append(
                (
                    os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images", upper_garment),
                    upper_annotation_pair[upper_garment],
                    os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/lower_body/images", lower_garment),
                    lower_annotation_pair[lower_garment],
                    # os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/test_ali", human)
                    os.path.join("/apdcephfs_cq8/share_1367250/yijicheng/data/ft_local/DressCode/upper_body/images", human)
                )
            )

    print(examples)

    upper_garm_list_path = []
    upper_garm_list_prompt = []
    lower_garm_list_path = []
    lower_garm_list_prompt = []
    human_list_path = []
    for example in examples:
        upper_garm_list_path.append(example[0])
        upper_garm_list_prompt.append(example[1])
        lower_garm_list_path.append(example[2])
        lower_garm_list_prompt.append(example[3])
        human_list_path.append(example[4])

    # print(examples)


    def pil_to_tensor(images):
        images = np.array(images).astype(np.float32) / 255.0
        images = torch.from_numpy(images.transpose(2, 0, 1)) # c h w
        return images

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_examples):
            super(DummyDataset, self).__init__()
            self.num_examples = num_examples
        def __getitem__(self, index):
            return index
        def __len__(self):
            return self.num_examples

    test_dataset = DummyDataset(len(examples))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4,
    )
    test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    use_face_mask = False
    if use_face_mask:
        facemasker = FaceMaskerV2(device=f"cuda:{device.index}")

    ##default human 
    # for i in range(len(human_list_path)):
    for i in test_dataloader:
        output_dir_ = os.path.join(output_dir, f"{human_list_path[i].split('/')[-1]}-{upper_garm_list_path[i].split('/')[-1]}-{lower_garm_list_path[i].split('/')[-1]}")
        os.makedirs(output_dir_, exist_ok=True)
        human_img=Image.open(human_list_path[i]).convert("RGB")
        upper_garm_img=Image.open(upper_garm_list_path[i]).convert("RGB")
        lower_garm_img=Image.open(lower_garm_list_path[i]).convert("RGB")

        # lower_garm_img = resize_and_paste(lower_garm_img)

        upper_prompt=upper_garm_list_prompt[i]
        lower_prompt=lower_garm_list_prompt[i]
        is_checked=True
        is_checked_crop=False
        denoise_steps=30
        seed=None
        
        if use_face_mask:
            face_mask = get_face_mask(np.array(human_img), facemasker) # h w c
            face_mask = torch.tensor(face_mask).permute(2, 0, 1)

        # for image_idx in range(4):
        #     image_out_upper, masked_img_upper = start_tryon(human_img, upper_garm_img, upper_prompt, is_checked, is_checked_crop, denoise_steps, seed, 0)

        #     image_out_upper.resize(human_img.size).save(os.path.join(output_dir, 'out_hd_upper' + '_' + str(image_idx) + '.png'))
        #     masked_img_upper.resize(human_img.size).save(os.path.join(output_dir, "masked_img_upper.png"))


        #     image_out, masked_img = start_tryon(image_out_upper, lower_garm_img, lower_prompt, is_checked, is_checked_crop, denoise_steps, seed, 1)


        #     image_out.resize(human_img.size).save(os.path.join(output_dir, 'out_hd_upper_lower' + '_' + str(image_idx) + '.png'))
        #     masked_img.resize(human_img.size).save(os.path.join(output_dir, "masked_img_upper_lower.png"))
            
        for image_idx in range(4):
            image_out_lower, mask_lower, masked_img_lower, pose_img_pil_lower = start_tryon(human_img, lower_garm_img, lower_prompt, is_checked, is_checked_crop, denoise_steps, seed, 1)
            image_out, mask, masked_img, pose_img_pil = start_tryon(image_out_lower, upper_garm_img, upper_prompt, is_checked, is_checked_crop, denoise_steps, seed, 0)

            gt_human_sample = pil_to_tensor(human_img.resize(human_img.size))
            gt_lower_sample = pil_to_tensor(lower_garm_img.resize(human_img.size))
            gt_upper_sample = pil_to_tensor(upper_garm_img.resize(human_img.size))

            image_out_lower = pil_to_tensor(image_out_lower.resize(human_img.size))
            image_out = pil_to_tensor(image_out.resize(human_img.size))

            if use_face_mask:
                image_out_lower = image_out_lower * ~face_mask + gt_human_sample * face_mask
                image_out = image_out * ~face_mask + gt_human_sample * face_mask

            use_bg_mask = True
            if use_bg_mask:
                mask_lower = torch.tensor(np.array(mask_lower.resize(human_img.size))  / 255, dtype=torch.bool)[None]
                mask = torch.tensor(np.array(mask.resize(human_img.size))  / 255, dtype=torch.bool)[None]
                mask = mask | mask_lower

                image_out_lower = image_out_lower * mask_lower + gt_human_sample * ~mask_lower
                image_out = image_out * mask + gt_human_sample * ~mask

            pose_img_lower = pil_to_tensor(pose_img_pil_lower.resize(human_img.size))
            pose_img = pil_to_tensor(pose_img_pil.resize(human_img.size))

            # print(gt_human_sample.shape, gt_lower_sample.shape, image_out_lower.shape, gt_upper_sample.shape, image_out.shape)
            x_sample = torch.cat([gt_human_sample, pose_img_lower, gt_lower_sample, image_out_lower, pose_img, gt_upper_sample, image_out], dim=2)
            torchvision.utils.save_image(x_sample, os.path.join(output_dir_, 'out_hd_lower_upper' + '_' + str(image_idx) + '.png'))

            # masked_img_lower.resize(human_img.size).save(os.path.join(output_dir_, 'masked_img_lower' + '_' + str(image_idx) + '.png'))
            # masked_img.resize(human_img.size).save(os.path.join(output_dir_, 'masked_img_upper_lower' + '_' + str(image_idx) + '.png'))

            # pose_img_pil_lower.resize(human_img.size).save(os.path.join(output_dir_, 'pose_img_lower' + '_' + str(image_idx) + '.png'))
            # pose_img_pil.resize(human_img.size).save(os.path.join(output_dir_, 'pose_img_upper_lower' + '_' + str(image_idx) + '.png'))

if __name__ == '__main__':
    main()