#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
    python3 onnx_inference.py --model=./../../weights/onnx/11/rtcdet_p.onnx --image_path=./test.jpg --num_classes=2
"""
import argparse
import os

import cv2
import time
import numpy as np
import sys
sys.path.append('../../')

import onnxruntime
from utils.misc import PreProcessor, PostProcessor, multiclass_nms
from utils.vis_tools import visualize


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument("-m", "--model", type=str, default="../../weights/onnx/11/yolov1.onnx",
                        help="Input your onnx model.")
    parser.add_argument("-i", "--image_path", type=str, default='../test_image.jpg',
                        help="Path to your input image.")
    parser.add_argument("-o", "--output_dir", type=str, default='../../det_results/onnx/',
                        help="Path to your output directory.")
    parser.add_argument("-s", "--score_thr", type=float, default=0.35,
                        help="Score threshould to filter the result.")
    parser.add_argument("-size", "--img_size", type=int, default=640,
                        help="Specify an input shape for inference.")
    parser.add_argument("-class", "--num_classes", type=int, default=80,
                        help="The number classess of the model.")
    return parser


## Post-processer
class PostProcessor1(object):
    def __init__(self, num_classes, conf_thresh=0.15, nms_thresh=0.5):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh


    def __call__(self, bboxes, scores, labels):
        """
        Input:
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        print("1111111111")
        print(type(bboxes))
        #bboxes = bboxes[..., :4]
        #scores = scores[..., 4:]
        print(type(scores), bboxes.shape, scores.shape, labels.shape)
        bboxes = np.squeeze(bboxes, 0)
        scores = np.squeeze(scores, 0)
        labels = np.squeeze(labels, 0)
        print(type(scores), bboxes.shape, scores.shape, labels.shape)

        # scores & labels
        labels = np.argmax(scores, axis=1)                      # [M,]
        scores = scores[(np.arange(scores.shape[0]), labels)]   # [M,]

        # thresh
        keep = np.where(scores > self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)
        print("2222222222")

        return bboxes, scores, labels


if __name__ == '__main__':
    args = make_parser().parse_args()

    # class color for better visualization
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]

    # preprocessor
    prepocess = PreProcessor(img_size=args.img_size)

    # postprocessor1
    postprocess = PostProcessor1(num_classes=args.num_classes, conf_thresh=args.score_thr, nms_thresh=0.5)

    # read an image
    input_shape = tuple([args.img_size, args.img_size])
    origin_img = cv2.imread(args.image_path)

    # preprocess
    x, ratio = prepocess(origin_img)

    t0 = time.time()
    # inference
    session = onnxruntime.InferenceSession(args.model)

    print(session.get_inputs()[0].name)
    ort_inputs = {session.get_inputs()[0].name: x[None, :, :, :]}
    output = session.run(None, ort_inputs)
    print("inference time: {:.1f} ms".format((time.time() - t0)*1000))

    t0 = time.time()
    # post process
    print(type(output), len(output), type(output[0]), output[0].shape, output[1].shape, output[2].shape)
    #bboxes, scores, labels = postprocess(output[0])
    bboxes, scores, labels = postprocess(output[0], output[1], output[2])
    bboxes /= ratio
    print("post-process time: {:.1f} ms".format((time.time() - t0)*1000))

    # visualize detection
    origin_img = visualize(
        img=origin_img,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        vis_thresh=args.score_thr,
        class_colors=class_colors
        )

    # show
    #cv2.imshow('onnx detection', origin_img)
    #cv2.waitKey(0)

    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
