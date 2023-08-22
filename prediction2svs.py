#!/usr/bin/env python
# coding: utf-8

# In[31]:


import json
import openslide

import colorsys

from tifffile import imsave
import os

import cv2
import math
import random
import colorsys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm

def visualize_instances_dict(
    input_image, inst_dict, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.

    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        
        type_colour: a dict of {type_id : (type_name, colour)} , 
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    overlay = np.copy((input_image))

    #inst_rng_colors = random_colors(len(inst_dict))
    #inst_rng_colors = np.array(inst_rng_colors) * 255
    #inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        #print(inst_info)
        inst_contour = np.array(inst_info["contour"]).reshape((-1,1,2)).astype(np.int32)
        #np.array(inst_info["contour"]).astype(np.float32)
        #print(idx)
        #print(inst_info["type"])
        #if inst_info["type"] == 0:
            #print(inst_info["type"])
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            inst_colour = (inst_rng_colors[idx]).tolist()
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)
    
        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay
    
    
    
file_name = 'TCGA-4T-AA8H-01A-01-TSA.17AD2FB9-4E87-42BE-81A1-5E0035137749'
json_files = 'wsi_out/json/' + file_name + '.json'

with open(json_files, 'r') as j:
     results = json.loads(j.read())
     
     
slide = openslide.open_slide('test/wsi/' + file_name + '.svs')
[m, n] = slide.dimensions
image = np.array(slide.read_region((0,0),0,(m,n)).convert('RGB')).astype(np.uint8)
print(int(slide.properties['openslide.objective-power']))


#inst_map = np.load('test/inst_pred/' +file_name+ '.npy')
#print(inst_map.shape)


type_info_dict = json.load(open('type_info.json', "r"))
type_info_dict = {
    int(k): (v[0], tuple(v[1])) for k, v in type_info_dict.items()
}
print(type_info_dict)
overlay_result = visualize_instances_dict(image,results['nuc'],type_colour=type_info_dict)




overlay_result = overlay_result.astype(np.uint8)
np.save('TCGA-4T-AA8H-01A-01-TSA.17AD2FB9-4E87-42BE-81A1-5E0035137749.npy',overlay_result)
overlay_result = np.load('TCGA-4T-AA8H-01A-01-TSA.17AD2FB9-4E87-42BE-81A1-5E0035137749.npy').astype(np.uint8)
imsave(file_name+'.tif', overlay_result)
import pyvips
image = pyvips.Image.new_from_file('TCGA-4T-AA8H-01A-01-TSA.17AD2FB9-4E87-42BE-81A1-5E0035137749.tif')
image.tiffsave("TCGA-4T-AA8H-01A-01-TSA.17AD2FB9-4E87-42BE-81A1-5E0035137749_pyramid.tiff", tile=True, compression='jpeg',pyramid=True, bigtiff=True)


