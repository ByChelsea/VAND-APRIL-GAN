import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import json
import open_clip
from model import LinearLayer
from prompt_ensemble import encode_text_with_prompt_ensemble

def test_single_image(image_path, model_path, config_path, output_path):
    img_size = 518
    features_list = [6, 12, 18, 24]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', img_size, pretrained='openai')
    model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')

    with open(config_path, 'r') as f:
        model_configs = json.load(f)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), 'ViT-L-14-336').to(device)
    checkpoint = torch.load(model_path)
    linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    obj_list = ['object']
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features, patch_tokens = model.encode_image(image_tensor, features_list)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = text_prompts['object'].unsqueeze(0)
        
        # 像素级异常检测
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
        
        anomaly_map = np.sum(anomaly_maps, axis=0)
    
    anomaly_map = anomaly_map[0]
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    
    image_resized = cv2.resize(np.array(image), (img_size, img_size))
    anomaly_map_resized = cv2.resize(anomaly_map, (img_size, img_size))
    anomaly_map_resized = (anomaly_map_resized * 255).astype(np.uint8)
    anomaly_map_color = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)
    anomaly_map_color = cv2.cvtColor(anomaly_map_color, cv2.COLOR_BGR2RGB)
    
    result = cv2.addWeighted(image_resized, 0.5, anomaly_map_color, 0.5, 0)
    
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, 'anomaly_map.png'), anomaly_map_resized)
    cv2.imwrite(os.path.join(output_path, 'result.png'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    print(f"Anomaly score (max): {anomaly_map.max():.4f}")
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/home/ps/few-shot-research/AdaCLIP/test_image/006.jpg', 
                       help='Path to the input image (default: ./test_image.jpg)')
    parser.add_argument('--model_path', type=str, 
                       default='/home/ps/few-shot-research/VAND-APRIL-GAN/exps/pretrained/visa_pretrained.pth',
                       help='Path to the trained model (default: ./exps/pretrained/mvtec_pretrained.pth)')
    parser.add_argument('--config_path', type=str, 
                       default='/home/ps/few-shot-research/VAND-APRIL-GAN/open_clip/model_configs/ViT-L-14-336.json',
                       help='Path to model config (default: ./open_clip/model_configs/ViT-L-14-336.json)')
    parser.add_argument('--output_path', type=str, default='./results', 
                       help='Path to save results (default: ./results/custom)')
    args = parser.parse_args()
    
    test_single_image(args.image_path, args.model_path, args.config_path, args.output_path)