"""
    python3 export_onnx.py --model=rtcdet_p --num_classes=2 --dynamic --weight=./../weights/plate/rtcdet_p/rtcdet_p_bs256_best_2023-09-27_06-09-12.pth

    python3 export_onnx.py --model=yolox_n --num_classes=2 --dynamic --weight=./../weights/plate/yolox_n/yolox_n_best.pth
"""
"""
import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from yolox.exp import get_exp
from yolox.utils import replace_module
from yolox.models.network_blocks import SiLU
"""
import argparse
import os, sys
import warnings
from loguru import logger
sys.path.append('..')

import torch
from torch import nn

from utils.misc import SiLU
from utils.misc import load_weight, replace_module

from config import  build_model_config
from models.detectors import build_model
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser("YOLO ONNXRuntime")
    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument("--input", default="images", type=str,
                        help="input node name of onnx model")
    parser.add_argument("--output", default="output", type=str,
                        help="output node name of onnx model")
    parser.add_argument("-o", "--opset", default=11, type=int,
                        help="onnx opset version")
    parser.add_argument("--batch", type=int, default=1,
                        help="batch size")
    parser.add_argument("--dynamic", action="store_true", default=False,
                        help="whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", default=False,
                        help="use onnxsim or not")
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="experiment description file")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument('--save_dir', default='../weights/onnx/', type=str,
                        help='Dir to save onnx file')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument('-nc', '--num_classes', default=80, type=int,
                        help='topk candidates for testing')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')
    parser.add_argument('--nms_class_agnostic', action='store_true', default=False,
                        help='Perform NMS operations regardless of category.')
    return parser


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.size())
        #x = x.transpose(1, 2)
        #print(x.dim())
        boxes = x[..., :4]
        scores, classes = torch.max(x[..., 4:], 1, keepdim=True)
        classes = classes.float()
        print(boxes.size(), scores.size(), classes.size())
        """
        boxes = x[:, :, :4]
        objectness = x[:, :, 4:5]
        scores, classes = torch.max(x[:, :, 5:], 2, keepdim=True)
        scores *= objectness
        classes = classes.float()
        """
        return boxes, scores, classes


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolox_export(weights, exp_file):
    exp = get_exp(exp_file)
    model = exp.get_model()
    ckpt = torch.load(weights, map_location='cpu')
    model.eval()
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True
    return model, exp


def main():
    args = parse_args().parse_args()
    suppress_warnings()

    logger.info("args value: {}".format(args))
    print('\nStarting: %s' % args.weight)

    print('Opening YOLOX model')

    device = torch.device('cpu')


    ##model, exp = yolox_export(args.weights, args.exp)

    device = torch.device('cpu')

    # Dataset & Model Config
    model_cfg = build_model_config(args)

    # build model
    model = build_model(args, model_cfg, device, args.num_classes, False, deploy=True)

    # replace nn.SiLU with SiLU
    model = replace_module(model, nn.SiLU, SiLU)

    # load trained weight
    model = load_weight(model, args.weight, args.fuse_conv_bn)

    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device).eval()

    #model.float()
    #for k, m in model.named_modules():
    #    print(m)


    model = nn.Sequential(model, DeepStreamOutput())

    #img_size = [exp.input_size[1], exp.input_size[0]]
    img_size = [args.img_size, args.img_size]

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)

    #onnx_output_file = os.path.basename(args.weights).split('.pt')[0] + '.onnx'
    # save onnx file
    save_path = os.path.join(args.save_dir, str(args.opset))
    os.makedirs(save_path, exist_ok=True)
    output_name = os.path.join(args.model + '.onnx')
    onnx_output_file = os.path.join(save_path, output_name)

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'boxes': {
            0: 'batch'
        },
        'scores': {
            0: 'batch'
        },
        'classes': {
            0: 'batch'
        }
    }

    print('Exporting the model to ONNX')
    torch.onnx.export(
        model, 
        onnx_input_im, 
        onnx_output_file, 
        #verbose=False, 
        opset_version=args.opset,
        #do_constant_folding=True, 
        input_names=['input'], 
        output_names=['boxes', 'scores', 'classes'],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )
    logger.info("generated onnx model named {}".format(onnx_output_file))

    if True:
    #if args.simplify:
        print('Simplifying the ONNX model')
        import onnx
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('Done: %s\n' % onnx_output_file)


"""
def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOX conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-c', '--exp', required=True, help='Input exp (.py) file path (required)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if not os.path.isfile(args.exp):
        raise SystemExit('Invalid exp file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args
"""


if __name__ == '__main__':
    sys.exit(main())