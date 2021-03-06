CUDA_VISIBLE_DEVICES=0 python train.py \
  --backbone resnet_adv\
  --lr 0.05 \
  --workers 1 \
  --epochs 500 \
  --batch-size 8 \
  --ratio 3 \
  --gpu-ids 0 \
  --checkname MCubeSNet \
  --eval-interval 1 \
  --loss-type ce \
  --dataset multimodal_dataset \
  --list-folder list_folder \
  --use-pretrained-resnet \
  --is-multimodal \
  --use-nir \
  --use-aolp \
  --use-dolp
