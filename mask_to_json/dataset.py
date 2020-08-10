import os
import shutil
import random
import numpy as np

# This function creates json file from image masks

def split(main_dir, images_dir, masks_dir, val_ratio, test_ratio):

    # Getting the root directory
    root_dir = os.path.dirname(main_dir)

    # Creating paths for 3 new directories in Dataset
    train_dir = os.path.join(os.path.join(root_dir, "dataset"), "train")
    val_dir = os.path.join(os.path.join(root_dir, "dataset"), "val")
    test_dir = os.path.join(os.path.join(root_dir, "dataset"), "test")
    
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    # Creating images and masks folder inside train, val and test folders
    train_img_dir = os.path.join(train_dir, "images")
    val_img_dir = os.path.join(val_dir, "images")
    test_img_dir = os.path.join(test_dir, "images")
    os.makedirs(train_img_dir)
    os.makedirs(val_img_dir)
    os.makedirs(test_img_dir)

    train_mask_dir = os.path.join(train_dir, "masks")
    val_mask_dir = os.path.join(val_dir, "masks")
    test_mask_dir = os.path.join(test_dir, "masks")
    os.makedirs(train_mask_dir)
    os.makedirs(val_mask_dir)
    os.makedirs(test_mask_dir)

    # Get all files from images folder and randomize order  
    all_files = os.listdir(images_dir)
    np.random.shuffle(all_files)

    # Splitting up files
    train_files, test_files = np.split(np.array(all_files), [int(len(all_files)* (1 - test_ratio))])
    train_files, val_files = np.split(train_files, [int(len(train_files)* (1 - val_ratio))])
    train_files = train_files.tolist()
    val_files   = val_files.tolist()
    test_files  = test_files.tolist()

    # Copying files from MainData to Dataset 
    for train_file in train_files:
        orig_image_dir = os.path.join(images_dir, train_file)
        orig_mask_dir = os.path.join(masks_dir, train_file)
        shutil.copy(orig_image_dir, train_img_dir)
        shutil.copy(orig_mask_dir, train_mask_dir)

    for val_file in val_files:
        orig_image_dir = os.path.join(images_dir, val_file)
        orig_mask_dir = os.path.join(masks_dir, val_file)
        shutil.copy(orig_image_dir, val_img_dir)
        shutil.copy(orig_mask_dir, val_mask_dir)

    for test_file in test_files:
        orig_image_dir = os.path.join(images_dir, test_file)
        orig_mask_dir = os.path.join(masks_dir, test_file)
        shutil.copy(orig_image_dir, test_img_dir)
        shutil.copy(orig_mask_dir, test_mask_dir)

    return len(all_files), len(train_files), len(val_files), len(test_files)
