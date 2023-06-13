import torch
from dataset import VisaDataset, MVTecDataset

def memory(model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot, few_shot_features,
           dataset_name, device):
    normal_features_ls = {}
    for i in range(len(obj_list)):
        if dataset_name == 'mvtec':
            normal_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                       aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path,
                                       obj_name=obj_list[i])
        else:
            normal_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                      mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj_list[i])
        normal_dataloader = torch.utils.data.DataLoader(normal_data, batch_size=1, shuffle=False)
        image_features_ls = []
        for items in normal_dataloader:
            image = items['img'].to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                patch_tokens = [p[0, 1:, :] for p in patch_tokens]
                image_features_ls.append(patch_tokens)
        normal_features_ls[obj_list[i]] = [torch.cat([image_features_ls[j][i] for j in range(len(image_features_ls))],
                                                     dim=0) for i in range(len(image_features_ls[0]))]
    return normal_features_ls