# -------------------------- Train YOLO7 series --------------------------
# 3080ti,  ulimit -n 2048
python3 train.py \
        --cuda \
        -d plate \
        --root /home/david/dataset/lpd_lpr/detect_plate_datasets_coco \
        -m yolov7_tiny \
        -bs 256 \
        -size 160 \
        --num_workers 8 \
        --wp_epoch 3 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --no_aug_epoch 20 \
        --grad_accumulate 1 \
        --ema \
        --fp16 \
        --mixup 0.0 \
        --load_cache \
        --multi_scale > ./logs/matster.log &
        # --load_cache \
        # --resume weights/coco/yolox_m/yolox_m_best.pth \
        # --eval_first

# -------------------------- Train YOLOX & YOLOv7 series --------------------------
# python train.py \
#         --cuda \
#         -d coco \
#         --root /data/datasets/ \
#         -m yolox_s \
#         -bs 8 \
#         -size 640 \
#         --wp_epoch 3 \
#         --max_epoch 300 \
#         --eval_epoch 10 \
#         --no_aug_epoch 15 \
#         --grad_accumulate 8 \
#         --ema \
#         --fp16 \
#         --multi_scale \
#         # --load_cache \
#         # --resume weights/coco/yolox_m/yolox_m_best.pth \
#         # --eval_first

# -------------------------- Train YOLOv1~v5 series --------------------------
# python train.py \
#         --cuda \
#         -d coco \
#         --root /mnt/share/ssd2/dataset/ \
#         -m yolov5_s \
#         -bs 16 \
#         -size 640 \
#         --wp_epoch 3 \
#         --max_epoch 300 \
#         --eval_epoch 10 \
#         --no_aug_epoch 10 \
#         --ema \
#         --fp16 \
#         --multi_scale \
#         # --load_cache \
#         # --resume weights/coco/yolov5_l/yolov5_l_best.pth \
#         # --eval_first
