# 3080ti,  ulimit -n 2048


# -------------------------- Train YOLOX series --------------------------
# yolox_n
python3 train.py \
        --cuda \
        -d plate \
        --root /home/david/dataset/lpd_lpr/detect_plate_datasets_coco \
        -m yolox_n \
        -bs 128 \
        -size 320 \
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


# -------------------------- Train YOLO7 series --------------------------
# yolo7_tiny
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


# -------------------------- Train rtcdev series --------------------------
# rtcdet_p
python3 train.py \
        --cuda \
        -d plate \
        --root /home/david/dataset/lpd_lpr/detect_plate_datasets_coco \
        -m rtcdet_p \
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


python3 train.py \
        --cuda \
        -d plate \
        --root /home/david/dataset/lpd_lpr/detect_plate_datasets_coco \
        -m rtcdet_p \
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
        --multi_scale > ./logs/matster.log &


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