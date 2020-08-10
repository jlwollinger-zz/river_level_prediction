"""
Created on Sat Fri 8 20:03:51 2020

@author: Wollinger
"""

import sys
import json
import datetime
import numpy as np
import time
import imgaug  
import os
import skimage.draw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = "./coco_weights/mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./model_logs"

############################################################
#  Configurations
############################################################


class RiverConfig(Config):
    """Configuration for training on River Dataset.
    Derives from the base Config class and overrides values specific
    to the River dataset.
    """
    # Give the configuration a recognizable name
    NAME = "river"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class RiverDataset(utils.Dataset):
    def load_river(self, dataset_dir, subset):
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train, val or test
        """
        self.add_class("river", 1, "river")

        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
      
        annotations1 = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations1.values())

        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
        
            image_path = os.path.join(dataset_dir, a['filename'])
            height = a['height']
            width = a['width']

            self.add_image(
                "river",  
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)


    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "river":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "river":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    dataset_train = RiverDataset()
    dataset_train.load_river(dataset, "train")
    dataset_train.prepare()

    dataset_val = RiverDataset()
    dataset_val.load_river(dataset, "val")
    dataset_val.prepare()

    history = model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')
    
    
    history = model.keras_model.history.history
    
    print(history.keys())
    
    best_epoch = np.argmin(history["val_loss"]) 
    print("Best Epoch:", best_epoch + 1, history["val_loss"][best_epoch])


   
############################################################
#  Training
############################################################

if __name__ == '__main__':

    dataset = './dataset2'
    weights = COCO_WEIGHTS_PATH
    logs = DEFAULT_LOGS_DIR
    weights_path = COCO_WEIGHTS_PATH

    # Configurations
    config = RiverConfig()
    #config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs)


    weights_path = COCO_WEIGHTS_PATH


    weights_path = model.find_last()
    
    # Load weights
    print("Loading weights ", weights_path)
    
        # Exclude the last layers because they require a matching
        # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    
    model.load_weights(weights_path, by_name=True)

    train(model)