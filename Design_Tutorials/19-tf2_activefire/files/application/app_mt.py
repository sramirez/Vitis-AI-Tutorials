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

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

divider = '------------------------------------'


def preprocess_fn(image_path, fix_scale):
    '''
    Image pre-processing.
    Rearranges from BGR to RGB then normalizes to range 0:1
    and then scales by input quantization scaling factor
    input arg: path of image file
    return: numpy array
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image * (1/255.0) * fix_scale
    image = image.astype(np.int8)
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def run_dpu(id, start, dpu, imgs):

    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    #output_fixpos = outputTensors[0].get_attr("fix_point")
    #output_scale = 1 / (2**output_fixpos)

    batch_size = input_ndim[0]
    n_of_images = len(imgs)
    count = 0
    ids = []
    ids_max = 50
    output_data = []
    for i in range(ids_max):
        output_data.append([np.empty(output_ndim, dtype=np.int8, order="C")])

    while count < n_of_images:
        if count+batch_size <= n_of_images:
            run_size = batch_size
        else:
            run_size = n_of_images - count

        '''prepare batch input/output '''
        input_data = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(run_size):
            image_run = input_data[0]
            image_run[j, ...] = imgs[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(input_data, output_data[len(ids)])
        ids.append((job_id, run_size, start+count))
        count = count + run_size
        if count < n_of_images:
            if len(ids) < ids_max-1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            '''store output vectors '''
            for j in range(ids[index][1]):
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(output_data[0][j] * output_scale)
                out_q[write_index] = np.argmax(output_data[index][0][j])
                write_index += 1
        ids = []


def app(image_dir, threads, model):

    listimage = os.listdir(image_dir)
    run_total = len(listimage)

    global out_q
    out_q = [None] * run_total
    g = xir.Graph.deserialize(model)
    print("hola2")
    subgraphs = get_child_subgraph_dpu(g)

    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    print("hola3")

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    ''' preprocess images '''
    print(divider)
    print('Pre-processing', run_total, 'images...')
    img = []
    for i in range(run_total):
        path = os.path.join(image_dir, listimage[i])
        img.append(preprocess_fn(path, input_scale))

    '''run threads '''
    print('Starting', threads, 'threads...')
    thread_all = []
    start = 0
    for i in range(threads):
        if i == threads - 1:
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=run_dpu, args=(i, start, all_dpu_runners[i], in_q))
        thread_all.append(t1)
        start = end

    time1 = time.time()
    for x in thread_all:
        x.start()
    for x in thread_all:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(run_total / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (fps, run_total, timetotal))

    ''' post-processing '''
    classes = ['dog', 'cat']
    correct = 0
    wrong = 0
    print('Post-processing', len(out_q), 'images..')
    for i in range(len(out_q)):
        prediction = classes[out_q[i]]
        ground_truth, _ = listimage[i].split('.', 1)
        if ground_truth == prediction:
            correct += 1
        else:
            wrong += 1
    accuracy = correct/len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' % (correct, wrong, accuracy))
    print(divider)
    
    return


# only used if script is run as 'main' from command line
def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')
    ap.add_argument('-t', '--threads', type=int, default=1, help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model', type=str, default='customcnn.xmodel',
                    help='Path of xmodel. Default is customcnn.xmodel')

    args = ap.parse_args()

    print('Command line options:')
    print(' --image_dir : ', args.image_dir)
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)

    app('../data/patches', args.threads, '../build/compiled_vck190/customcnn.xmodel')


if __name__ == '__main__':
    main()
