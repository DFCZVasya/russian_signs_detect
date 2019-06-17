#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
from small_nn import small_predict as sp
import tsv
from PIL import Image
import csv

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
   help='path to an image or an video (mp4 format)')

def read_file(DIR, count, filename, writer, yolo, config, model_sm):
    image = cv2.imread(DIR)
    print(DIR)
    boxes = yolo.predict(image)
    if len(boxes) != 0:
        image_h, image_w, _ = image.shape
        image2 = Image.open(DIR)
        for box in boxes:
            cl = draw_boxes(image, image2, box, config['model']['labels'], count, model_sm)
            xmin = int(box.xmin * image_w)
            ymin = int(box.ymin * image_h)
            xmax = int(box.xmax * image_w)
            ymax = int(box.ymax * image_h)
            file_name = "2018_02-13_1418_left/" + filename[:-4]
            if cl != '0':
                #writer.list_line([file_name, xmin, ymin, xmax, ymax, cl])
                writer.writerow([file_name, xmin, ymin, xmax, ymax, cl])
    print(len(boxes), 'boxes are found')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################
    model_sm = sp.load_small_model()
    yolo.load_weights('full_yolo_sign_full_dataset.h5')

    ###############################
    #   Predict bounding boxes 
    ###############################
    DIR = 'image2/'
    count = 0
    with open('output.tsv', 'wt') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(["frame", "xtl", "ytl", "xbr", "ybr", "class"])
   # writer = tsv.TsvWriter(open("answers.tsv", "w"))
       # writer.line("frame", "xtl", "ytl", "xbr", "ybr", "class")
        filenames = os.listdir(DIR)
        filenames.sort()
        for filename in filenames:
            if filename.endswith('.jpg'):
                read_file(DIR + filename, count, filename, writer, yolo, config, model_sm)
                count += 1



    #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
