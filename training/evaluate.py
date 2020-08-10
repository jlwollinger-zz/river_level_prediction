# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:55:39 2020

@author: Wollinger
"""

from mrcnn import visualize
import mrcnn.model as modellib
from skimage.io import imread
from mrcnn.config import Config
import matplotlib.pyplot as plt
import scrap_live_stream
import scrap_mesurament
import load_defaut_image
import level_calculator

from PIL import Image
import cv2
import numpy
from skimage.color import rgb2gray
from skimage.filters import threshold_sauvola
from datetime import datetime
import threading
from flask import Flask
from flask import jsonify

#from sklearn.externals import joblib
#regressor = joblib.load('regressor.sav')


class EvaluateConfig(Config):
    NAME = "river"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # River Dataset has 1 class
    #STEPS_PER_EPOCH = 100
    #VALIDATION_STEPS = 50
    DETECTION_MIN_CONFIDENCE = 0.9
    #GPU_COUNT = 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
        
config = EvaluateConfig()

model_dir = './trained_model//mask_rcnn_river.h5'

model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)

RIVER_WEIGHTS_PATH = './trained_model/mask_rcnn_river.h5'

model.load_weights(model_dir, by_name=True)

default_mask = load_defaut_image.load()#88_215

app = Flask(__name__)

results_scrapped = []
results_calculated = []
last_level_calculated = 0
last_level_scrapped = 0
last_hour = datetime.now()

secondThread = threading.Thread(target = start)
secondThread.start()   

@app.route('/level')
def getLevel():
    return jsonify(
            level = last_level_calculated,
            level_scrapped= last_level_scrapped,
            hour = last_hour
            )


app.run()

def start():
    while(True):
        global last_level_calculated
        global last_level_scrapped
        global last_hour
        levels = []
        levels_calculated = []
        current_level = []
        
        for i in range(10):
            img = scrap_live_stream.getLiveStreamImageFrame()
        
            
        
            try:
                result = model.detect([img], verbose=1)[0] #First class
            except:
                continue
        
            masked_image = visualize.display_instances(img, result['rois'], result['masks'], result['class_ids'], 'river',
                                                    show_bbox=False, show_mask=True,  colors = [(0, 0.8, 0.7)])
            
            
            #plt.figure()
            #plt.imshow(img2)
            #plt.imsave('MASKED-FINAL.png', masked_image)
        
            
            POINTS_TO_TRAVERSE = [150, 200, 250, 300, 400, 450, 550, 600, 650, 700]
        
            for point in POINTS_TO_TRAVERSE:    
               cv2.line(masked_image, (0, point), (len(img[0]), point), (255,0,0), 2)
            plt.imshow(masked_image)
            #plt.imsave('traced-image.png', img2)
        
            mask = result['masks']
        
                
            level = level_calculator.calculateLevel(default_mask, mask)
            levels.append(level)
            
            _, level, _ = scrap_mesurament.findFirstWaterLevel()
            current_level.append(level)
            
            
            
        levels = level_calculator.remove_standard_deviation_single(levels)
        levels.sort()
        mediana = 0
        mediana = levels[int(len(levels) / 2)]
        print(levels)
        
        results_calculated.append(mediana)
        
        results_scrapped.append(current_level[0])
        
        last_level_calculated = mediana
        last_hour = datetime.now()
        last_level_scrapped = current_level[0]
        
        print('scraped')
        print(results_scrapped)
        print('ordinary')
        print(results_calculated)
    
    
