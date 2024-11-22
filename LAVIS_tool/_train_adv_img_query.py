import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
import wandb

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess

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

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


# to replace vis_processor
transform_a = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Resize(size=(384, 384), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        # torchvision.transforms.ToTensor(),
    ]
)
transform_b = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        # torchvision.transforms.ToTensor(),
    ]
)
normalize = torchvision.transforms.Compose(
    [   
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


def _i2t(args, txt_processors, model, image):
    
    # normalize image here
    image = normalize(image / 255.0)
    
    # generate caption
    if args.model_name == "img2prompt_vqa":
        question = "what is the content of this image?"
        question = txt_processors["eval"](question)
        samples  = {"image": image, "text_input": [question] * (image.size()[0])}
        
        # obtain gradcam and update dict
        samples  = model.forward_itm(samples=samples)
        samples  = model.forward_cap(samples=samples, num_captions=1, num_patches=20)
        caption  = samples['captions']
        for cap_idx, cap in enumerate(caption):
            if cap_idx == 0:
                caption_merged = cap
            else:
                caption_merged = caption_merged + cap
    else:
        samples  = {"image": image}
        caption  = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
        caption_merged = caption
    
    return caption_merged


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    # load models for i2t
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--input_res", default='224', type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--output", default="temp", type=str)
    parser.add_argument("--data_path", default="temp", type=str)
    parser.add_argument("--text_path", default="temp.txt", type=str)
    
    parser.add_argument("--delta", default="normal", type=str)
    parser.add_argument("--num_query", default=20, type=int)
    parser.add_argument("--num_sub_query", default=5, type=int)
    parser.add_argument("--sigma", default=16, type=float)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    
    args = parser.parse_args()

    # ---------------------- #
    print(f"Loading LAVIS models: {args.model_name}, model_type: {args.model_type}...")
    # load models for i2t
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    
    # use clip text coder for attack
    clip_img_model_rn50,   _ = clip.load("RN50", device=device, jit=False)
    clip_img_model_rn101,  _ = clip.load("RN101", device=device, jit=False)
    clip_img_model_vitb16, _ = clip.load("ViT-B/16", device=device, jit=False)
    clip_img_model_vitb32, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_img_model_vitl14, _ = clip.load("ViT-L/14", device=device, jit=False)
    print("Done")
    # ---------------------- #

    # load clip_model params
    num_sub_query, num_query, sigma = args.num_sub_query, args.num_query, args.sigma
    batch_size    = args.batch_size
    alpha         = args.alpha
    epsilon       = args.epsilon
    
    if args.model_name == 'blip2_opt':
        args.input_res = 224
    else:
        args.input_res = 384

    if args.input_res == 384:
        vit_adv_data  = ImageFolderWithPaths(args.data_path, transform=transform_a)
        clean_data    = ImageFolderWithPaths("path to imagenet-val", transform=transform_a)
    
    else:
        vit_adv_data  = ImageFolderWithPaths(args.data_path, transform=transform_b)
        clean_data    = ImageFolderWithPaths("path to imagenet-val", transform=transform_b)
        
    data_loader       = torch.utils.data.DataLoader(vit_adv_data, batch_size=batch_size, shuffle=False, num_workers=24)
    clean_data_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=24)
        
    # org text/features
    adv_vit_text_path = args.text_path
    with open(os.path.join(adv_vit_text_path), 'r') as f:
        lavis_text_of_adv_vit  = f.readlines()[:args.num_samples] 
        f.close()
    
    with torch.no_grad():
        adv_vit_text_token    = clip.tokenize(lavis_text_of_adv_vit).to(device)
        adv_vit_text_features = clip_img_model_vitb32.encode_text(adv_vit_text_token)
        adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
        adv_vit_text_features = adv_vit_text_features.detach()
    
    # tgt text/features
    tgt_text_path = 'path to _coco_captions_10000.txt'
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:args.num_samples] 
        f.close()
    
    # clip text features of the target
    with torch.no_grad():
        target_text_token    = clip.tokenize(tgt_text).to(device)
        target_text_features = clip_img_model_vitb32.encode_text(target_text_token)
        target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
        target_text_features = target_text_features.detach()

    # baseline results
    vit_attack_results   = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    query_attack_results = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    assert (vit_attack_results == query_attack_results).all()
    
    ## other arch
    with torch.no_grad():
        # rn50
        adv_vit_text_features_rn50 = clip_img_model_rn50.encode_text(adv_vit_text_token)
        adv_vit_text_features_rn50 = adv_vit_text_features_rn50 / adv_vit_text_features_rn50.norm(dim=1, keepdim=True)
        adv_vit_text_features_rn50 = adv_vit_text_features_rn50.detach()
        target_text_features_rn50  = clip_img_model_rn50.encode_text(target_text_token)
        target_text_features_rn50  = target_text_features_rn50 / target_text_features_rn50.norm(dim=1, keepdim=True)
        target_text_features_rn50  = target_text_features_rn50.detach()
        vit_attack_results_rn50    = torch.sum(adv_vit_text_features_rn50 * target_text_features_rn50, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_rn50  = torch.sum(adv_vit_text_features_rn50 * target_text_features_rn50, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_rn50 == query_attack_results_rn50).all()

        # rn101
        adv_vit_text_features_rn101 = clip_img_model_rn101.encode_text(adv_vit_text_token)
        adv_vit_text_features_rn101 = adv_vit_text_features_rn101 / adv_vit_text_features_rn101.norm(dim=1, keepdim=True)
        adv_vit_text_features_rn101 = adv_vit_text_features_rn101.detach()
        target_text_features_rn101  = clip_img_model_rn101.encode_text(target_text_token)
        target_text_features_rn101  = target_text_features_rn101 / target_text_features_rn101.norm(dim=1, keepdim=True)
        target_text_features_rn101  = target_text_features_rn101.detach()
        vit_attack_results_rn101    = torch.sum(adv_vit_text_features_rn101 * target_text_features_rn101, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_rn101  = torch.sum(adv_vit_text_features_rn101 * target_text_features_rn101, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_rn101 == query_attack_results_rn101).all()

        # vitb16
        adv_vit_text_features_vitb16 = clip_img_model_vitb16.encode_text(adv_vit_text_token)
        adv_vit_text_features_vitb16 = adv_vit_text_features_vitb16 / adv_vit_text_features_vitb16.norm(dim=1, keepdim=True)
        adv_vit_text_features_vitb16 = adv_vit_text_features_vitb16.detach()
        target_text_features_vitb16  = clip_img_model_vitb16.encode_text(target_text_token)
        target_text_features_vitb16  = target_text_features_vitb16 / target_text_features_vitb16.norm(dim=1, keepdim=True)
        target_text_features_vitb16  = target_text_features_vitb16.detach()
        vit_attack_results_vitb16    = torch.sum(adv_vit_text_features_vitb16 * target_text_features_vitb16, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_vitb16  = torch.sum(adv_vit_text_features_vitb16 * target_text_features_vitb16, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_vitb16 == query_attack_results_vitb16).all()

        # vitl14
        adv_vit_text_features_vitl14 = clip_img_model_vitl14.encode_text(adv_vit_text_token)
        adv_vit_text_features_vitl14 = adv_vit_text_features_vitl14 / adv_vit_text_features_vitl14.norm(dim=1, keepdim=True)
        adv_vit_text_features_vitl14 = adv_vit_text_features_vitl14.detach()
        target_text_features_vitl14  = clip_img_model_vitl14.encode_text(target_text_token)
        target_text_features_vitl14  = target_text_features_vitl14 / target_text_features_vitl14.norm(dim=1, keepdim=True)
        target_text_features_vitl14  = target_text_features_vitl14.detach()
        vit_attack_results_vitl14    = torch.sum(adv_vit_text_features_vitl14 * target_text_features_vitl14, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_vitl14  = torch.sum(adv_vit_text_features_vitl14 * target_text_features_vitl14, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_vitl14 == query_attack_results_vitl14).all()
    ## ----------
    
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, reinit=True)
    
    for i, ((image, _, path), (image_clean, _, _)) in enumerate(zip(data_loader, clean_data_loader)):
        if batch_size * (i+1) > args.num_samples:
            break
        image = image.to(device)  # size=(10, 3, 224, 224)
        image_clean = image_clean.to(device)  # size=(10, 3, 224, 224)
        
        # obtain all text features (via CLIP text encoder)
        adv_text_features = adv_vit_text_features[batch_size * (i): batch_size * (i+1)]        
        tgt_text_features = target_text_features[batch_size * (i): batch_size * (i+1)]
        
        # ------------------- random gradient-free method
        print("init delta with diff(adv-clean)")
        delta = torch.tensor((image - image_clean))
        torch.cuda.empty_cache()
        
        best_caption = lavis_text_of_adv_vit[i]
        better_flag = 0
        
        for step_idx in range(args.steps):
            print(f"{i}-th image - {step_idx}-th step")
            # step 1. obtain purturbed images
            if step_idx == 0:
                image_repeat      = image.repeat(num_query, 1, 1, 1)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)
            else:
                image_repeat      = adv_image_in_current_step.repeat(num_query, 1, 1, 1)             

                lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
                adv_vit_text_token_in_current_step      = clip.tokenize(lavis_text_of_adv_image_in_current_step).to(device)
                adv_vit_text_features_in_current_step   = clip_img_model_vitb32.encode_text(adv_vit_text_token_in_current_step)
                adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step / adv_vit_text_features_in_current_step.norm(dim=1, keepdim=True)
                adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step.detach()                
                adv_text_features     = adv_vit_text_features_in_current_step
                torch.cuda.empty_cache()
            query_noise            = torch.randn_like(image_repeat).sign() # Rademacher noise
            perturbed_image_repeat = torch.clamp(image_repeat + (sigma * query_noise), 0.0, 255.0)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)
            
            # num_query is obtained via serveral iterations
            text_of_perturbed_imgs = []
            for query_idx in range(num_query//num_sub_query):
                sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]
                if args.model_name == 'img2prompt_vqa':
                    text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat)
                else:
                    with torch.no_grad():
                        text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat)
                text_of_perturbed_imgs.extend(text_of_sub_perturbed_imgs)
            
            # step 2. estimate grad
            with torch.no_grad():
                perturb_text_token    = clip.tokenize(text_of_perturbed_imgs).to(device)
                perturb_text_features = clip_img_model_vitb32.encode_text(perturb_text_token)
                perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
                perturb_text_features = perturb_text_features.detach()
            
            coefficient = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)  # size = (num_query * batch_size)
            coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise = query_noise.reshape(num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = coefficient * query_noise / sigma # size = (num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = pseudo_gradient.mean(0) # size = (bs, 3, args.input_res, args.input_res)
            
            # step 3. log metrics
            delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
            delta.data = delta_data
            print(f"img: {i:3d}-step {step_idx} max  delta", torch.max(torch.abs(delta)).item())
            print(f"img: {i:3d}-step {step_idx} mean delta", torch.mean(torch.abs(delta)).item())
            
            adv_image_in_current_step = torch.clamp(image_clean+delta, 0.0, 255.0)
            
            # get adv text
            if args.model_name == 'img2prompt_vqa':
                lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
            else:
                with torch.no_grad():
                    lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
            
            # log sim
            with torch.no_grad():
                lavis_text_token = clip.tokenize(lavis_text_of_adv_image_in_current_step).to(device)
                lavis_text_features_of_adv_image_in_current_step = clip_img_model_vitb32.encode_text(lavis_text_token)
                lavis_text_features_of_adv_image_in_current_step = lavis_text_features_of_adv_image_in_current_step / lavis_text_features_of_adv_image_in_current_step.norm(dim=1, keepdim=True)
                lavis_text_features_of_adv_image_in_current_step = lavis_text_features_of_adv_image_in_current_step.detach()

                adv_txt_tgt_txt_score_in_current_step = torch.mean(torch.sum(lavis_text_features_of_adv_image_in_current_step * tgt_text_features, dim=1)).item()
                
                # update results
                if adv_txt_tgt_txt_score_in_current_step > query_attack_results[i]:
                    query_attack_results[i] = adv_txt_tgt_txt_score_in_current_step
                    best_caption = lavis_text_of_adv_image_in_current_step[0]
                    better_flag  = 1
                    
                # other clip archs
                # rn50
                tgt_text_features_rn50 = target_text_features_rn50[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn50 = clip_img_model_rn50.encode_text(lavis_text_token)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50 / text_features_of_adv_image_in_current_step_rn50.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50.detach()
                adv_txt_tgt_txt_score_in_current_step_rn50 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn50 * tgt_text_features_rn50, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn50 > query_attack_results_rn50[i]:
                    query_attack_results_rn50[i] = adv_txt_tgt_txt_score_in_current_step_rn50
                
                # rn101
                tgt_text_features_rn101 = target_text_features_rn101[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn101 = clip_img_model_rn101.encode_text(lavis_text_token)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101 / text_features_of_adv_image_in_current_step_rn101.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101.detach()
                adv_txt_tgt_txt_score_in_current_step_rn101 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn101 * tgt_text_features_rn101, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn101 > query_attack_results_rn101[i]:
                    query_attack_results_rn101[i] = adv_txt_tgt_txt_score_in_current_step_rn101
                
                # vitb16
                tgt_text_features_vitb16 = target_text_features_vitb16[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitb16 = clip_img_model_vitb16.encode_text(lavis_text_token)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16 / text_features_of_adv_image_in_current_step_vitb16.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16.detach()
                adv_txt_tgt_txt_score_in_current_step_vitb16 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitb16 * tgt_text_features_vitb16, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitb16 > query_attack_results_vitb16[i]:
                    query_attack_results_vitb16[i] = adv_txt_tgt_txt_score_in_current_step_vitb16
                
                # vitl14
                tgt_text_features_vitl14 = target_text_features_vitl14[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitl14 = clip_img_model_vitl14.encode_text(lavis_text_token)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14 / text_features_of_adv_image_in_current_step_vitl14.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14.detach()
                adv_txt_tgt_txt_score_in_current_step_vitl14 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitl14 * tgt_text_features_vitl14, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitl14 > query_attack_results_vitl14[i]:
                    query_attack_results_vitl14[i] = adv_txt_tgt_txt_score_in_current_step_vitl14
                    # ----------------
                

        if args.wandb:
            wandb.log(
                {   
                    "moving-avg-adv-rn50"    : np.mean(vit_attack_results_rn50[:(i+1)]),
                    "moving-avg-query-rn50"  : np.mean(query_attack_results_rn50[:(i+1)]),
                    
                    "moving-avg-adv-rn101"   : np.mean(vit_attack_results_rn101[:(i+1)]),
                    "moving-avg-query-rn101" : np.mean(query_attack_results_rn101[:(i+1)]),
                    
                    "moving-avg-adv-vitb16"  : np.mean(vit_attack_results_vitb16[:(i+1)]),
                    "moving-avg-query-vitb16": np.mean(query_attack_results_vitb16[:(i+1)]),
                    
                    "moving-avg-adv-vitb32"  : np.mean(vit_attack_results[:(i+1)]),
                    "moving-avg-query-vitb32": np.mean(query_attack_results[:(i+1)]),
                    
                    "moving-avg-adv-vitl14"  : np.mean(vit_attack_results_vitl14[:(i+1)]),
                    "moving-avg-query-vitl14": np.mean(query_attack_results_vitl14[:(i+1)]),
                }
            )

        # log text
        print("best caption of current image:", best_caption)
        with open(os.path.join(args.output + '.txt'), 'a') as f:
            # print(''.join([best_caption]), file=f)
            if better_flag:
                f.write(best_caption+'\n')
            else:
                f.write(best_caption)
        f.close()