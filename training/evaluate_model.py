import os
import sys
import random
import math
import re
import time
import imageio
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from skimage.io import imread

import river
import argparse
TK_SILENCE_DEPRECATION=1


############################################################
#  Evaluating
############################################################

if __name__ == '__main__':
    import argparse


    MODEL_DIR = './trained_model/mask_rcnn_river.h5'
    RIVER_WEIGHTS_PATH = './trained_model/mask_rcnn_river.h5'
    RIVER_DIR = './to_test'
    RESULT_DIR = './results'

    config = river.RiverConfig()

    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    # Load test dataset
    print("Loading Test Set.......")
    dataset = river.RiverDataset()
    dataset.load_river(RIVER_DIR, "test")

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load the last model you trained
    weights_path = RIVER_WEIGHTS_PATH

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    
    img = imread("C:\\Users\\Wollinger\\Desktop\\TCC\\0.43_2008.jpg")
    
    result = model.detect([img], verbose=1)[0] #First class
    
    masked_image = visualize.display_instances(img, result['rois'], result['masks'], result['class_ids'], 'river',
                                        show_bbox=False, show_mask=True,  colors = [(0, 0.8, 0.7)])
    plt.imshow(masked_image)
    
    # Going through all images and saving their masks
    pbar = tqdm(dataset.image_ids)
    for image_id in pbar:

        pbar.set_description("Segmenting Images")

        info = dataset.image_info[image_id]
        
        # Run model
        image = dataset.load_image(image_id)
        results = model.detect([image], verbose=0)
        r = results[0]

        # Save results
        result_path = os.path.join(RESULT_DIR, info["id"])
        masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names,
                                        show_bbox=False, show_mask=True,  colors = [(0, 0.8, 0.7)])
        imageio.imwrite(result_path, masked_image)
        pbar.update(1)