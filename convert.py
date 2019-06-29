ann_dir = "/home/lawrence/PycharmProjects/test/maskrcnn-benchmark/datasets/seafloor/annotations/subset-20_ann"
orig_dir = "/home/lawrence/PycharmProjects/test/maskrcnn-benchmark/datasets/seafloor/subset-20"
config_dir = "/home/lawrence/PycharmProjects/test/maskrcnn-benchmark/datasets/seafloor/config.json"

from PIL import Image
import os, sys
from random import randint
import numpy
from copy import deepcopy
import json

from skimage.measure import find_contours


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


def flood_fill(filled, imageArr, color, origColor, origx, origy):
    s = Stack()
    s.push([origx, origy])
    while(not s.isEmpty()):
        x, y = s.pop()
        if(x >= len(imageArr) or x < 0 or y >= len(imageArr[0]) or y < 0 or filled[x][y] or not imageArr[x][y] == origColor):
            continue
        filled[x][y] = True
        imageArr[x][y] = color
        s.push([x, y + 1])
        s.push([x + 1, y])
        s.push([x - 1, y])
        s.push([x, y - 1])


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
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

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


for imageFile in os.listdir(orig_dir):
    print("step")
    #check if it is the mask image
    if "mask.png" in imageFile and not "water" in imageFile and not "color" in imageFile:

        image = Image.open(orig_dir + "/" + imageFile)
        #convert to 2d array
        imageArr = numpy.array(image)
        imageArr = imageArr[:, :, 0]
        copy = deepcopy(imageArr)

        #flood fill to find individual objects and change their color
        filled = [[False for i in range(len(imageArr[0]))] for j in range(len(imageArr))]

        colors = []
        colors_location = []

        for x in range(len(imageArr)):
            for y in range(len(imageArr[0])):

                if(not filled[x][y]):
                    color = randint(0, 10000)
                    flood_fill(filled, copy, color, imageArr[x][y], x, y)
                    colors.append(color)
                    colors_location.append([x, y])

        #make polygons
        polygons = []
        for color in colors:
            polygons.append(find_contours(imageArr, color))

        #convert to json
        imageJSON = {}
        imageJSON["license"] = 0
        imageJSON["file_name"] = imageFile
        imageJSON["height"], imageJSON["width"] = image.size
        imageJSON["id"] = imageFile
        dataset["images"].append(imageJSON)

        annotation = {}
        annotation["segmentation"] = []







