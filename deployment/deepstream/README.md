
# Generate onnx file

```
$ python3 export_onnx.py --model=rtcdet_p --num_classes=2 --dynamic --weight=./../../weights/plate/rtcdet_p/rtcdet_p_bs256_best_2023-09-27_06-09-12.pth --img_size=160
```


# Inference by onnx file

```
$ python3 onnx_inference.py --mode=dir --output_dir=./results --model=./../../weights/onnx/11/rtcdet_p.onnx --image_path=./test --num_classes=2 --img_size=160
```




