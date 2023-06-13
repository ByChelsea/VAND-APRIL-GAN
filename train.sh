### train on the MVTec AD dataset
python train.py --dataset mvtec --train_data_path ./data/mvtec \
--save_path ./exps/visa/vit_large_14_518 --config_path ./open_clip/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
--features_list 6 12 18 24 --pretrained openai --image_size 518  --batch_size 8 --aug_rate 0.2 --print_freq 1 \
--epoch 3 --save_freq 1


### train on the VisA dataset
python train.py --dataset visa --train_data_path ./data/visa \
--save_path ./exps/mvtec/vit_large_14_518 --config_path ./open_clip/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
--features_list 6 12 18 24 --pretrained openai --image_size 518  --batch_size 8 --print_freq 1 \
--epoch 15 --save_freq 1

