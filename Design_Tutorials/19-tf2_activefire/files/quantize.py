'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Quantize the floating-point model
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import rasterio
import os
from glob import glob
from models import *
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np


MANUAL_ANNOTATION_PATH = 'data/manual_annotations_patches'
IMAGES_PATH = 'data/landsat_patches'
CUDA_DEVICE = 0
# 10 or 3
N_CHANNELS = 3
# 16 or 64
N_FILTERS = 16
MASK_ALGORITHM = 'Voting'
MODEL_NAME = 'unet'
MASKS_ALGORITHMS = ['Schroeder', 'Murphy', 'Kumar-Roy', 'Intersection', 'Voting']
# Desconsideras as mascaras que contem os seguintes itens no nome
IGNORE_MASKS_WITH_STR = ['v2']
# Remove as strings definidas do nome das mascas para realizar um match entre mascara e imagem analisada
REMOVE_STR_FROM_MASK_NAME = MASKS_ALGORITHMS + ['v1']
TH_FIRE = 0.25
IMAGE_SIZE = (256, 256)
MAX_PIXEL_VALUE = 65535  # Max. pixel value, used to normalize the image
OUTPUT_DIR = './log'

DIVIDER = '-----------------------------------------'


def load_path_as_dataframe(mask_path):
    masks = glob(os.path.join(mask_path, '*.tif'))

    print('Carregando diretório: {}'.format(mask_path))
    print('Total de máscaras no diretórios: {}'.format(len(masks)))

    df = pd.DataFrame(masks, columns=['masks_path'])
    # recupera o nome da máscara pelo caminho dela
    df['original_name'] = df.masks_path.apply(os.path.basename)
    # remove o algoritmo gerador da mascara do nome dela
    df['image_name'] = df.original_name.apply(remove_algorithms_name)

    # remove mascaras com as strings definidas
    for ignore_mask_with_str in IGNORE_MASKS_WITH_STR:
        df = df[~df.original_name.str.contains(ignore_mask_with_str)]

    return df


def remove_algorithms_name(mask_name):
    """Remove o nome dos algoritmos do nome da máscara"""

    # algorithms_name = MASKS_ALGORITHMS + ['mask' , 'noFire']
    # algorithms_name = MASKS_ALGORITHMS + ['v2' , 'noFire']

    for algorithm in REMOVE_STR_FROM_MASK_NAME:
        mask_name = mask_name.replace('_{}'.format(algorithm), '')

    return mask_name


def merge_dataframes(df_manual, df_images):
    return pd.merge(df_manual, df_images, on='image_name', how='left')


def open_manual_annotation(path):
    if type(path) != str:
        mask = np.zeros(IMAGE_SIZE)
    else:
        mask = get_mask_arr(path)

    return np.array(mask, dtype=np.uint8)


def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]


def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE

    return img


def get_img_762bands(path):
    img = rasterio.open(path).read((7, 6, 2)).transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE

    return img


def load_dataset(images_path=IMAGES_PATH):
    # Define a função de abertura de imagens
    open_image = get_img_arr
    if N_CHANNELS == 3:
        open_image = get_img_762bands

    df_manual = load_path_as_dataframe(MANUAL_ANNOTATION_PATH)
    df_images = load_path_as_dataframe(images_path)

    df = pd.merge(df_manual, df_images, on='image_name', how='left')

    print('Total de Imagens: {}'.format(len(df.index)))

    data = []
    for index, row in tqdm(df.iterrows()):
        data.append((open_image(row['masks_path_y']), open_manual_annotation(row['masks_path_x'])))

    return data


def statistics3 (y_true, y_pred):
    y_pred_neg = 1 - y_pred
    y_expected_neg = 1 - y_true

    tp = np.sum(y_pred * y_true)
    tn = np.sum(y_pred_neg * y_expected_neg)
    fp = np.sum(y_pred * y_expected_neg)
    fn = np.sum(y_pred_neg * y_true)
    return tn, fp, fn, tp


def evaluate_model(model, test_dataset):
    y_pred_all_v1 = []
    y_true_all_v1 = []

    for (img, mask) in test_dataset:
        y_pred = model.predict(np.array([img]), batch_size=1)
        y_true = mask[:, :] > TH_FIRE
        y_pred = y_pred[0, :, :, 0] > TH_FIRE
        y_pred_all_v1.append(y_pred)
        y_true_all_v1.append(y_true)

    y_pred_all_v1 = np.array(y_pred_all_v1, dtype=np.uint8).flatten()
    y_true_all_v1 = np.array(y_true_all_v1, dtype=np.uint8).flatten()
    tn, fp, fn, tp = statistics3(y_true_all_v1, y_pred_all_v1)
    P = float(tp) / (tp + fp)
    R = float(tp) / (tp + fn)
    IoU = float(tp) / (tp + fp + fn)
    F = (2 * P * R) / (P + R)
    print('P: :', P, ' R: ', R, ' IoU: ', IoU, ' F-score: ', F)
    return F


def quant_model(float_path, quant_path, evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    float_model = get_model(MODEL_NAME,
                            input_height=IMAGE_SIZE[0],
                            input_width=IMAGE_SIZE[1],
                            n_filters=N_FILTERS,
                            n_channels=N_CHANNELS)
    float_model.load_weights(float_path)
    print('Weights Loaded')

    ## QUANTIZATION!!
    # make folder for saving quantized model
    head_tail = os.path.split(quant_path)
    os.makedirs(head_tail[0], exist_ok=True)

    # make TFRecord dataset and image processing pipeline
    #quant_dataset = input_fn_quant(tfrec_dir, batchsize, height, width)
    test_dataset = load_dataset(images_path=IMAGES_PATH)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=np.stack([x[0] for x in test_dataset]))

    # saved quantized model
    quantized_model.save(quant_path)
    print('Saved quantized model to', quant_path)

    if evaluate:
        print('\n'+DIVIDER)
        print('Evaluating quantized model..')
        print(DIVIDER+'\n')
        evaluate_model(quantized_model, test_dataset)
        print('\n'+DIVIDER)
        print('Evaluating float model..')
        print(DIVIDER+'\n')
        evaluate_model(float_model, test_dataset)

    return


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_path',  type=str, default='weights/model_unet_Voting_final_weights.h5', help='Full path of floating-point model. Default is build/float_model/k_model.h5')
    ap.add_argument('-q', '--quant_path',  type=str, default='weights/quantized_activefire_model.h5', help='Full path of quantized model. Default is build/quant_path/q_model.h5')
    ap.add_argument('-e', '--evaluate', action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ', tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_path)
    print (' --quant_path  : ', args.quant_path)
    print (' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')

    quant_model(args.float_path, args.quant_path, args.evaluate)


if __name__ ==  "__main__":
    main()
