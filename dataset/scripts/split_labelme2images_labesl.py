#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    When the images and labels in the same directory, this file can split it into images dir and labels dir in train and val dir.

    $ python3 split_labelme2images_labesl.py --data_dir=./../echo_park --save_dir=./

    Then it can create train and val directory.
"""
import os, sys, signal, argparse
from typing import List, OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_dir', default='./data',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--train_ratio', default=0.8,type=float, help="Train ratio, val ratio is 1-train_ratio")

arg = parser.parse_args()


def term_sig_handler(signum, frame) -> None:
    sys.stdout.write('\r>> {}: catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    sys.stdout.flush()
    os._exit(0)


def get_file_list(input_dir:str, label_file_list:List[str])->None:
    imgs_list = []
    for (parent, dirnames, filenames) in os.walk(input_dir,  followlinks=True):
        for filename in filenames:
            if filename.split('.')[-1] == 'jpg':
                #print( os.path.join(parent, filename.split('.')[0]) )
                imgs_list.append( os.path.join(parent, filename.split('.')[0]) )
    #print(imgs_list)
    for (parent, dirnames, filenames) in os.walk(input_dir,  followlinks=True):
        for filename in filenames:
            if filename.split('.')[-1] == 'json':
                if os.path.join(parent, filename.split('.')[0]) in imgs_list:
                    label_file_list.append( os.path.join(parent, filename.split('.')[0]) )
    return


def split_labelme2images_labesl(arg)->None:
    print("Loading data from ", arg.data_dir)
    assert os.path.exists(arg.data_dir)
    train_labels_dir = os.path.join(arg.save_dir, 'train',  'labels')
    train_images_dir = os.path.join(arg.save_dir, 'train', 'images')
    if not os.path.exists(train_labels_dir):
        os.makedirs(train_labels_dir)
    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)
    val_labels_dir = os.path.join(arg.save_dir, 'val',  'labels')
    val_images_dir = os.path.join(arg.save_dir, 'val', 'images')
    if not os.path.exists(val_labels_dir):
        os.makedirs(val_labels_dir)
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir)
    # get files
    label_file_list = []
    get_file_list(arg.data_dir, label_file_list)
    print( len(label_file_list) )
    for (i, header_file) in enumerate(label_file_list):
        images_dir = train_images_dir
        labels_dir = train_labels_dir
        if (i % 10) >= (arg.train_ratio*10):
            images_dir = val_images_dir
            labels_dir = val_labels_dir
        header_file_list = header_file.split('/')
        save_file_name = header_file_list[-2] + '_' + header_file_list[-1] + '.jpg'
        #print(header_file, save_file_name)
        src_file = os.path.abspath(header_file + '.jpg')
        dst_file = os.path.join(images_dir, save_file_name)
        print(src_file, dst_file)
        os.symlink(src_file, dst_file)
        #
        save_file_name = header_file_list[-2] + '_' + header_file_list[-1] + '.json'
        #print(header_file, save_file_name)
        src_file = os.path.abspath(header_file + '.json')
        dst_file = os.path.join(labels_dir, save_file_name)
        print(src_file, dst_file)
        os.symlink(src_file, dst_file)
    return


if __name__ == "__main__":
    signal.signal(signal.SIGINT, term_sig_handler)
    split_labelme2images_labesl(arg)