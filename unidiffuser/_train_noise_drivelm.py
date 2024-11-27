import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
import json

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  


# ------------------- Image and Video Handling Functions -------------------
def extract_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="ViT-B/32", type=str)
    parser.add_argument("--epsilon", default=0.1, type=float)
    parser.add_argument("--steps", default=160, type=int)
    parser.add_argument("--cle_data_path", default='/home/beihang/wlu/adllm/DriveLM/challenge/llama_adapter_v2_multimodal7b', type=str, help='path of the clean images')
    parser.add_argument("--tgt_data_path", default='/home/beihang/wlu/vlmattack/AttackVLM/stable-diffusion/target_imgs/samples/00000.png', type=str, help='path of the target images')
    parser.add_argument('--json', default='/home/beihang/wlu/adllm/DriveLM/challenge/json/test_llama.json')
    args = parser.parse_args()
    
    # load clip_model params
    epsilon = args.epsilon
    alpha = 2 * epsilon / args.steps
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)

    # ------------- pre-processing images/text ------------- # 
    # preprocess images    
    preprocess_wo_no = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(clip_model.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.CenterCrop(clip_model.visual.input_resolution),
            torchvision.transforms.ToTensor(),
        ]
    )
    preprocess_normalize = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    # start attack      
    # (bs, c, h, w)
    image_tgt = Image.open(args.tgt_data_path).convert('RGB')
    
    # get tgt featutres
    with torch.no_grad():
        tgt_image_features = clip_model.encode_image(preprocess(image_tgt).unsqueeze(0).to(device))
        tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)

    # -------- get adv image -------- #
    with open(args.json, 'r') as f:
        llama_data = json.load(f)
    for item in llama_data:
        img_list = item["image"]
        for img in img_list:
            save_path = os.path.join('/home/beihang/wlu/adllm/DriveLM', f"attackvlm_noise{epsilon}", img)
            if os.path.exists(save_path):
                continue
            jpg_path = os.path.join(args.cle_data_path, img)
            image_org = Image.open(jpg_path).convert('RGB')
            image_org = preprocess_wo_no(image_org).unsqueeze(0).to(device)
            delta = torch.zeros_like(image_org, requires_grad=True)
            for j in range(args.steps):
                adv_image = image_org + delta
                adv_image = preprocess_normalize(adv_image).to(device)
                adv_image_features = clip_model.encode_image(adv_image)
                adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

                embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))  # computed from normalized features (therefore it is cos sim.)
                embedding_sim.backward()
                
                grad = delta.grad.detach()
                d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
                delta.data = d
                delta.grad.zero_()
                print(f"step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")

            # save imgs
            save_path = os.path.join('/home/beihang/wlu/adllm/DriveLM', f"attackvlm_noise{epsilon}", img)
            save_path = save_path.replace(".jpg", "-noise1.jpg")
            folder = os.path.dirname(save_path)
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            torchvision.utils.save_image(delta, save_path)
