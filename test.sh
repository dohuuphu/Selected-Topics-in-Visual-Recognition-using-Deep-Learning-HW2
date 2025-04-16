export CUDA_VISIBLE_DEVICES=0
cd demo/
python demo.py --config-file /mnt/SSD7/yuwei-hdd3/selected/HW2/detectron2/configs/HW2/X101_pretrained_ratio_3_deform_ciou_7res.yaml \
  --input /mnt/SSD7/yuwei-hdd3/selected/HW2/nycu-hw2-data/test \
  --output ../results/X101_pretrained_ratio_3_deform \
  --opts MODEL.WEIGHTS ../tools/exp_2/X-101_pretrained_ratio3_deform_ciou_7resX-101_pretrained/model_final.pth
