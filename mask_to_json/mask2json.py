from PIL import Image
import numpy as np                                 
from skimage import measure                        
from shapely.geometry import Polygon, MultiPolygon 
import glob
import json
import os
from tqdm import tqdm
import shutil


def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    sub_mask = np.array(sub_mask)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'bbox': bbox,
        'area': area
    }

    return annotation

def clean_up(data_dir):

    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')

    for filename in os.listdir(images_dir):
        image_file = os.path.join(images_dir, filename)
        new_image_file = os.path.join(data_dir, filename)
        shutil.move(image_file, new_image_file)
    
    shutil.rmtree(images_dir)
    shutil.rmtree(masks_dir)

def create_annotations(data_dir):

    images = []
    mask_paths = glob.glob(os.path.join(data_dir, "./masks/*.jpg"))

    mask_images = []
    pbar = tqdm(mask_paths)

    annotations = {}

    for mask_path in pbar:

        pbar.set_description("Creating Annotations")

        filename = mask_path.split('/')[-1]
        mask_image = Image.open(mask_path).convert('RGB')
        width, height = mask_image.size

        sub_masks = create_sub_masks(mask_image)
        base64_img_data = {}
        file_attributes = {}
        regions = {}

        id = 0
        for color, sub_mask in sub_masks.items():
            
            annotation = create_sub_mask_annotation(sub_mask)
            cords = annotation["segmentation"]
            my_cords = cords[0]

            shape_attributes = {}
            name = "polygon"
            all_x_points = my_cords[::2]
            all_y_points = my_cords[1::2] 
            assert len(all_x_points) == len(all_y_points)

            shape_attributes["name"] = name
            shape_attributes["all_points_x"] = all_x_points
            shape_attributes["all_points_y"] = all_y_points

            region_attributes = {}
            region_attributes["name"] = "river"      

            regions[str(id)] = {}
            regions[str(id)]["shape_attributes"] = shape_attributes
            regions[str(id)]["region_attributes"] = region_attributes

            id=id+1

        annotations[filename] = {}
        annotations[filename]["fileref"] = {}
        annotations[filename]["width"] = width
        annotations[filename]["height"] = height
        annotations[filename]["filename"] = filename
        annotations[filename]["base64_img_data"] = {}
        annotations[filename]["file_attributes"] = {}
        annotations[filename]["regions"] = regions
    
        pbar.update(1)


    json.dump(annotations, open(os.path.join(data_dir, "annotations.json"), "w"), indent=4)

    clean_up(data_dir)



