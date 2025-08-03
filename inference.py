import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

import open_clip
from model import LinearLayer
from dataset import VisaDataset, MVTecDataset
from prompt_ensemble import encode_text_with_prompt_ensemble


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value:
        return (pred - pred.min()) / (max_value - pred.min())
    elif max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def memory(model_name, model, cls_name, filenames, preprocess, few_shot_features, device):
    mem_features = {}
    for obj in cls_name:
        features = []
        for filename in filenames:
            image = Image.open(filename)
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                if 'ViT' in model_name:
                    patch_tokens = [p[0, 1:, :] for p in patch_tokens]
                else:
                    patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                features.append(patch_tokens)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features
    

def test_single_image(args):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_name = [args.cls_name]

    # Load the model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    
    # Load the image
    image = Image.open(args.image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    # Load the linear layer and checkpoint
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), args.model).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])

    # few shot
    if args.mode == 'few_shot':
        mem_features = memory(args.model, model, cls_name, args.filenames, preprocess, few_shot_features, device)


    # Perform inference on the single image
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features, patch_tokens = model.encode_image(image, features_list)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = []
        text_prompts = encode_text_with_prompt_ensemble(model, cls_name, tokenizer, device)
        for cls in cls_name:
            text_features.append(text_prompts[cls])
        text_features = torch.stack(text_features, dim=0)

        # pixel
        patch_tokens = linearlayer(patch_tokens)
        anomaly_maps = []
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_tokens[layer] @ text_features)
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=img_size, mode='bilinear', align_corners=True)
            anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
            anomaly_maps.append(anomaly_map.cpu().numpy())
        anomaly_map_ = anomaly_map = np.sum(anomaly_maps, axis=0)

        # few shot
        if args.mode == 'few_shot':
            image_features, patch_tokens = model.encode_image(image, few_shot_features)
            anomaly_maps_few_shot = []
            for idx, p in enumerate(patch_tokens):
                if 'ViT' in args.model:
                    p = p[0, 1:, :]
                else:
                    p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()
                cos = pairwise.cosine_similarity(mem_features[cls_name[0]][idx].cpu(), p.cpu())
                height = int(np.sqrt(cos.shape[1]))
                anomaly_map_few_shot = np.min((1 - cos), 0).reshape(1, 1, height, height)
                anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                        size=img_size, mode='bilinear', align_corners=True)
                anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
            anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
            anomaly_map = anomaly_map + anomaly_map_few_shot
            # anomaly_map = anomaly_map_few_shot
            # anomaly_map = normalize(anomaly_map_few_shot, max_value=3.0)
            
        # visualization
        save_vis = os.path.join(args.save_path, args.mode, 'imgs', args.model)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        path = [args.image_path]
        filename = path[0].split('/')[-1]
        jpg = "."+filename.split(".")[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        cv2.imwrite(os.path.join(save_vis, filename.replace(jpg, "")+'_ori'+jpg), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        mask = normalize(anomaly_map[0])
        # mask = anomaly_map[0]
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        cv2.imwrite(os.path.join(save_vis, filename), vis)
        # np.save(os.path.join(save_vis, filename.split(".")[0]+'_reference.npy'), anomaly_map_few_shot)
        # np.save(os.path.join(save_vis, filename.split(".")[0]+'_zero_shot.npy'), anomaly_map_)
        # np.save(os.path.join(save_vis, filename.split(".")[0]+'_few_shot.npy'), anomaly_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # paths
    parser.add_argument("--image_path", type=str, default="/home/acer/data/iphone15pro.PNG", help="path to test image")
    parser.add_argument("--save_path", type=str, default="/home/acer/VAND-APRIL-GAN/results/iphone/mvtec", help="path to test dataset")
    parser.add_argument("--cls_name", type=str, default="iphone", help="path to test dataset")
    parser.add_argument("--checkpoint_path", type=str, default='./exps/pretrained/mvtec_pretrained.pth', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    parser.add_argument("--filenames", type=str, nargs="+",
                        default=["/home/acer/data/my_photo-30.jpg", "/home/acer/data/pro_photo-7.jpg", "/home/acer/data/pro_photo-47.jpg", "/home/acer/data/iphone15pro.PNG"],
                        help="image paths for few shot")
    # few shot
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()

    setup_seed(args.seed)
    test_single_image(args)
