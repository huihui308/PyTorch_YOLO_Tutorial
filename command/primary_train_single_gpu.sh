# -------------------------- Train rtcdev series --------------------------
# rtcdet_p
python3 train.py \
        --cuda \
        --dataset traffic11 \
        --root /home/david/dataset/detect/yizhuang-COCO \
        -m rtcdet_p \
        -bs 256 \
        -size 640 \
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


# rtcdet_s
python3 train.py \
        --cuda \
        --dataset traffic11 \
        --root /home/david/dataset/detect/yizhuang-COCO \
        -m rtcdet_s \
        -bs 16 \
        -size 640 \
        --num_workers 8 \
        --wp_epoch 3 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --no_aug_epoch 20 \
        --grad_accumulate 1 \
        --ema \
        --fp16 \
        --mixup 0.5 \
        --multi_scale > ./logs/matster.log &


# rtcdet_x
python3 train.py \
        --cuda \
        --dataset traffic11 \
        --root /home/david/dataset/detect/yizhuang-COCO \
        -m rtcdet_x \
        -bs 4 \
        -size 640 \
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


python3 train.py \
        --cuda \
        --dataset traffic11 \
        --root /home/david/dataset/detect/yizhuang-COCO \
        -m rtcdet_x \
        -bs 4 \
        -size 640 \
        --num_workers 8 \
        --wp_epoch 3 \
        --max_epoch 300 \
        --eval_epoch 10 \
        --no_aug_epoch 20 \
        --grad_accumulate 1 \
        --ema \
        --fp16 \
        --mixup 0.0 \
        --resume weights/traffic11/rtcdet_x/rtcdet_x_bs4_best_2023-10-24_15-23-46.pth \
        --eval_first \
        --multi_scale > ./logs/matster.log &
