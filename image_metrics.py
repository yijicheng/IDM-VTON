import os
import torch
import torchvision
import numpy as np
from PIL import Image
from fire import Fire
from tqdm import tqdm, trange
from metrics.calculate_psnr import calculate_psnr
from metrics.calculate_ssim import calculate_ssim
from metrics.calculate_lpips import calculate_lpips
import json


import warnings

warnings.filterwarnings('ignore')


def compute_prefix_k(values, k):
    # 确保n不超过字典中元素的数量  
    k = min(k, len(values))  
    
    # 初始化前缀和  
    prefix_sum = 0  
    # 计算前n个元素的前缀和  
    for i in range(k):  
        prefix_sum += values[i]  
    
    # 计算平均值  
    average_prefix_sum = prefix_sum / k 
    return average_prefix_sum


def main(
    exp_dir,
    num_samples=5,
):
        batch_dir = sorted([x for x in sorted(os.listdir(os.path.join(exp_dir)))], key=lambda x: int(x.split("_")[0]))[:num_samples]
        print(batch_dir)
        sample = []
        target = []
        for batch in tqdm(batch_dir):
            image_path = os.path.join(exp_dir, batch)
            image = Image.open(image_path).convert("RGB")
            image = torchvision.transforms.functional.to_tensor(image) # CHW [0, 1]
            target.append(image[..., :768])
            sample.append(image[..., 768:])
            assert target[0].shape[2] == sample[0].shape[2]
            assert len(target) == len(sample)

        target = torch.stack(target, 0)[None] # 1NCHW
        sample = torch.stack(sample, 0)[None] # 1NCHW
        
        # result_psnr = calculate_psnr(sample, target)
        result_ssim = calculate_ssim(sample, target)
        result_lpips = calculate_lpips(sample, target, device=torch.device("cuda"))

        meta = {}
        for i, batch in enumerate(batch_dir):
            meta[batch] = {
                "SSIM": result_ssim["value"][i],
                "LPIPS": result_lpips["value"][i],
            }

        with open(os.path.join(exp_dir, "meta_results.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print("===================")
        print(f"[{exp_dir}]")
        # print(f"PSNR: {result_psnr['value']}")
        print(f"SSIM: {np.mean(list(result_ssim['value'].values())):.3f}")
        print(f"LPIPS: {np.mean(list(result_lpips['value'].values())):.3f}")
        print("===================")

if __name__ == '__main__':
    Fire(main)