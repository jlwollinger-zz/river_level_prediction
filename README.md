# River level measurment through deep learning with Mask-RCNN

This is a project created for the final assignment to graduate on the Information Technology bachelor degree.

It is a tool for measure the river level using Mask R-CNN and image processing. The following image sums up the project.



# Installation

Install Python 3

Run the following line in the same of the project directory.
```bash
pip install -r requirements.txt
```

The python scripts are arranged in the following order:

/scrapping
Contains the scprits to scrap the image and the river level of the Itajaí-Açú ir of the civil defense of Rio Do Sul-SC

/mask_to_json
This scrpits will separate the images of train, test and validation (dataset.py and annotate.py) and the script (mask2json.py) to extract the coordinates of the segmentation of a json file with the xy points.

/training
Contains the scripts for the training of the Mask R-CNN (river.py) and to measure the river level (evaluate.py and level_calculator.py)


# Usage
Use the scripts for scrap the images.
Following, use the script dataset.py and annotate.py to divide the images and afterwards use mask2json.py to extract the coordinates.
Use the script river.py to train the Mask R-CNN with the images divideds before.
Use the script evaluate.py and level_calculator.py to calibrate and measure the water level and make it available.
