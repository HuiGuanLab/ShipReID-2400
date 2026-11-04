CUDA_VISIBLE_DEVICES=2 python ../test.py \
--config_file ../configs/ship/vit_base.yml \
OUTPUT_DIR /media/disk2/wtz/ShipReID-2400-6257-上下左右匈牙利/exp/logs_上下左右分别对齐_top1_去遮蔽_PIL_单独1.5// \
DATASETS.NAMES "('ShipReID2400')" \
MODEL.PART_H 16 \
MODEL.PART_W 4 \
MODEL.PART_TRIPLET_LOSS_WEIGHT 5.0 \
TEST.WEIGHT '/media/disk2/wtz/ShipReID-2400-6257-上下左右匈牙利/exp/logs_上下左右分别对齐_top1_去遮蔽_PIL_单独1.5/ViT-B-16_300.pth'