export CUDA_VISIBLE_DEVICES=0

cd tools/
./train_net.py --num-gpus 1 --config-file ../configs/HW2/X101_pretrained_ratio_3_deform_ciou_7res.yaml

