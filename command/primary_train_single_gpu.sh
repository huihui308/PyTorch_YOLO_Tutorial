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