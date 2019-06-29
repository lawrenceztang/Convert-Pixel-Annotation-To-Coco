ann_dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/annotations/test.json"
orig_dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/test"
config_dir = "/home/lawrence/PycharmProjects/pytorch-mask-rcnn/seafloor/config.json"

from PIL import Image
import os, sys
from random import randint
import numpy
from copy import deepcopy
import json



def create_sub_masks(mask_image):
    width, height = mask_image.size
    for x in range(width):
        for y in range(height):
            supercategory = get_supercategory(mask_image.getpixel((x, y))[0])
            mask_image.putpixel((x, y), (supercategory, supercategory, supercategory))

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

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
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
        poly = poly.simplify(1.0)
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
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def get_rid_of_letter_add_1(imageFile):
    new_name = imageFile
    for i in range(len(imageFile) - 4):
        if not str.isdigit(new_name[i]):
            new_name = new_name[0:i] + new_name[i + 1:len(new_name)]
            i -= 1
    if new_name[0] == "0":
        new_name = 1 + new_name
    return new_name

#initialize format
dataset = {}
dataset["info"] = {}
dataset["licenses"] = []
dataset["images"] = []
dataset["categories"] = []
dataset["annotations"] = []

#get categories
config = json.load(open(config_dir))
labels = config["labels"]
for key, value in labels.items():
    temp = {}
    temp["supercategory"] = value["categorie"]
    temp["id"] = value["id"]
    temp["name"] = value["name"]
    dataset["categories"].append(temp)

annotation_id = 1
image_id = 1
is_crowd = 0
annotations = []

import os

def get_supercategory(id):
    # if 1 <= id <= 6 or id == 12:
    #     return numpy.int64(1)
    # elif 6 < id <= 10:
    #     return numpy.int64(2)
    # elif id == 11:
    #     return numpy.int64(3)

    if id == 1 or id == 2 or id == 3 or id == 4 or id == 5 or id == 6 or id == 12:
        return 1
    if id == 8 or id == 7:
        return 2
    if id == 10 or id == 9:
        return 3
    if id == 11:
        return 4
    if id == 0:
        return 0
    return id

for imageFile in os.listdir(orig_dir):

    #check if it is the mask image
    if "mask.png" in imageFile and not "water" in imageFile and not "color" in imageFile:

        image = Image.open(orig_dir + "/" + imageFile)


        #convert to json
        imageJSON = {}
        imageJSON["license"] = 0
        imageJSON["file_name"] = imageFile[:-9] + ".png"
        imageJSON["height"], imageJSON["width"] = image.size
        imageJSON["id"] = image_id
        imageJSON["date_captured"] = "0"
        imageJSON["url"] = "google.com"
        dataset["images"].append(imageJSON)


        # Create the annotations


        sub_masks = create_sub_masks(image)
        for color, sub_mask in sub_masks.items():
            category_id = (int)(color[1])
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
            annotation_id += 1
        image_id += 1



dataset["annotations"] = annotations

json.dump(dataset, open(ann_dir, "w"))

print("done")







