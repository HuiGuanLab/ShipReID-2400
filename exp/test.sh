CUDA_VISIBLE_DEVICES=2 python ../test.py \
--config_file ../configs/ship/vit_base.yml \
OUTPUT_DIR ./logs/ \
DATASETS.NAMES "('ShipReID2400')" \
MODEL.PART_H 16 \
MODEL.PART_W 4 \
TEST.WEIGHT ''