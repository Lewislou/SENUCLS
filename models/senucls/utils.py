import math
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from matplotlib import cm
import scipy
import cv2
from collections import Counter
from .positional_encodings import *
def get_infer_bboxes(inst_map,edge_num,point_num):

    obj_ids = torch.unique(inst_map)
    obj_ids = obj_ids[1:]
    h,w = inst_map.shape[0],inst_map.shape[1]
    inst_map = inst_map.reshape(inst_map.shape[0],inst_map.shape[1])
    binary = np.array(inst_map).copy()
    binary[binary>0] = 1
    
    centers = scipy.ndimage.center_of_mass(binary,inst_map,obj_ids)

    edge_points,edge_index = get_adjacent_matrix(centers,edge_num,'k-nearst')
    out_pos = positional_embedding(centers,h,w,scale=0.25)
    masks = inst_map == obj_ids[:, None, None]
    boxes = masks_to_boxes(masks)
    boxes = np.array(boxes)
    countour_points,select_shape_feats = get_countour_points(inst_map,obj_ids,centers,boxes,h,w,numbers=point_num)
    countour_points = countour_points.astype(float)
    boxes = torch.tensor(boxes)
    countour_points = torch.tensor(countour_points)
    select_shape_feats = torch.tensor(select_shape_feats)
    centers = torch.tensor(centers)
    edge_points = torch.tensor(edge_points)
    edge_index = torch.tensor(edge_index)
    edge_index = edge_index.permute(1,0)
    return boxes,centers,edge_points,edge_index,out_pos,select_shape_feats
    
def get_bboxes(inst_map,type_map,edge_num,point_num):

    obj_ids = torch.unique(inst_map)
    #print(obj_ids)
    obj_ids = obj_ids[1:]

    inst_map = inst_map.reshape(inst_map.shape[1],inst_map.shape[2])

    h,w =  inst_map.shape[0],inst_map.shape[1]
    binary = np.array(inst_map).copy()
    binary[binary>0] = 1
    
    centers = scipy.ndimage.center_of_mass(binary,inst_map,obj_ids)

    edge_points,edge_index = get_adjacent_matrix(centers,edge_num,'k-nearst')

    out_pos = positional_embedding(centers,h,w,scale=0.25)

    masks = inst_map == obj_ids[:, None, None]

    boxes = masks_to_boxes(masks)
    boxes = np.array(boxes)
    countour_points,select_shape_feats = get_countour_points(inst_map,obj_ids,centers,boxes,h,w,numbers=point_num)

    countour_points = countour_points.astype(float)

    classes = []
    for i in obj_ids:
        label = Counter(type_map[inst_map==i]).most_common(1)[0][0]-1
        classes.append(label)

    countour_points = torch.tensor(countour_points)
    boxes = torch.tensor(boxes)

    select_shape_feats = torch.tensor(select_shape_feats)
    centers = torch.tensor(centers)

    classes = torch.tensor(classes)

    edge_points = torch.tensor(edge_points)
    edge_index = torch.tensor(edge_index)
    edge_index = edge_index.permute(1,0)

    return boxes,classes,centers,edge_points,edge_index,out_pos,select_shape_feats
def pClosest(points, K,current):
 
    points.sort(key = lambda K: (K[0]-current[0])**2 + (K[1]-current[1])**2)
 
    return points[1:K+1]
def get_adjacent_matrix(centers,k,mode):
    N = len(centers)
    #print(centers)
    edge_points = []
    edge_index = []
    if N <= k:
        #print(N)
        all_indexs = [i for i in range(N)]

        for i in range(N):
            current = centers[i]
            adjans = list(set(all_indexs)-{i})

            for j in adjans:
                edge_index.append((i,j))
                edge_points.append(((current[0]+centers[j][0])/2,(current[1]+centers[j][1])/2))
        if len(edge_index) < 1:
             edge_index.append((0,0))

             edge_points.append((centers[0][0],centers[0][1]))
        edge_index = np.array(edge_index)
        return edge_points,edge_index
    for i in range(N):
        current = centers[i]

        k_nearest = pClosest(centers.copy(),k,current)

        for j in range(k):
            index = centers.index(k_nearest[j])

            edge_index.append((i,index))
            edge_points.append(((current[0]+k_nearest[j][0])/2,(current[1]+k_nearest[j][1])/2))

    edge_index = np.array(edge_index)
    
    #print(edge_index.shape)
    #print(edge_index[0])
    return edge_points,edge_index
        
def positional_embedding(centers,h,w,scale=0.25):
    p_enc_2d = PositionalEncoding2D(64)
    pos = torch.zeros((1,int(h*scale),int(w*scale),64))
    #print(h,w,pos.shape)
    pos_embed = p_enc_2d(pos)
    out_pos = []
    N = len(centers)
    for i in range(N):
        current = centers[i]
        out_pos.append(pos_embed[0,int(centers[i][0]*scale),int(centers[i][1]*scale)])
    out_pos = torch.stack(out_pos)
    return out_pos
        
        
def get_countour_points(inst_map,obj_ids,centers,boxes,h,w,numbers=4):
    inst_map = inst_map.reshape(h,w)
    select_contour = []
    select_shape_feats = []
    for i in range(len(obj_ids)):
        n_id = obj_ids[i]
        nuclei = np.zeros((h,w))
        nuclei[inst_map==n_id] = 255
        current_countour = []
        center = centers[i]
        bbox = boxes[i]
        for mask in [nuclei]:
            mask = np.uint8(mask)
            cnt, contour = get_single_centerpoint(mask)
            #print(cnt)
           
            contour = contour[0]
            contour = torch.Tensor(contour).float()
            x, y = center[1],center[0]
        select_contour.append((np.array(y), np.array(x)))
        current_countour.append((y,x))
        dists, coords = get_coordinates(x, y, contour,number=numbers)
        #print(dists, coords)
        point_index = [(0,0)]
        point_dist = [0]
        
        dists = dists.numpy()
        for j in range(len(dists)):
            point_index.append((0,j+1))
            point_dist.append(coords[int(360//numbers*j)])
            delt_y, delt_x = cv2.polarToCart(np.float64(coords[int(360//numbers*j)]), int(360//numbers*j), angleInDegrees = True)

            select_contour.append((y+delt_y[0],x+delt_x[0]))
            current_countour.append((y+delt_y[0],x+delt_x[0]))
            
        shape_feat = shape_feat_extract(current_countour,point_dist,point_index,bbox)
        select_shape_feats.append(shape_feat)
   
    countour_points = np.array(select_contour,dtype=object)
    select_shape_feats = np.array(select_shape_feats)

    return countour_points,select_shape_feats
    
    
def shape_feat_extract(delt_distance,point_dist,point_index,bbox,scale=0.25):
    max_h = bbox[3]+2//scale
    min_h = bbox[1]-2//scale
    max_w = bbox[2]+2//scale
    min_w = bbox[0]-2//scale
    p_enc_2d = single_PositionalEncoding2D(64)
    pos = torch.zeros((1,int((max_h-min_h)*scale),int((max_w-min_w)*scale),64))
    out_pos = []
    N = len(delt_distance)

    for i in range(N):
        pos_embddings = p_enc_2d((delt_distance[i][0]-min_h)*scale,(delt_distance[i][1]-min_w)*scale,pos).numpy()
        pos_embddings = np.append(pos_embddings,np.array(point_index[i]))
        pos_embddings = np.append(pos_embddings,np.array(point_dist[i]*scale))
        out_pos.append(pos_embddings)

    return out_pos    
    
def add_class(inst_map,classes):
    obj_ids = torch.unique(inst_map)
    obj_ids = obj_ids[1:].numpy()

    output = np.zeros((inst_map.shape[0],inst_map.shape[1]))
    for i in range(len(obj_ids)):
        output[inst_map==obj_ids[i]] = classes[i]+1

    return output
####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)

#class_weights = torch.tensor([2,7,3,7,20]).to("cuda")
#class_weights = torch.tensor([2,2,60,60]).to("cuda")
#class_weights = torch.tensor([2,3,10,11,21]).to("cuda")


def focal_loss(outputs, targets,class_weights,alpha=1, gamma=2,logits=False, reduce=True):
    class_weights = torch.tensor(class_weights).to("cuda")
    criterion1 = nn.CrossEntropyLoss(weight=class_weights,reduction='none',label_smoothing=0.1)
    ce_loss = criterion1(outputs, targets)
    #print('ce_loss',ce_loss)
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()

    if reduce:
        return torch.mean(focal_loss)
    else:
        return focal_loss

####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    #print(true.shape,pred.shape)
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss

####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss

def get_centerpoint(lis):
    area = 0.0
    x, y = 0.0, 0.0
    a = len(lis)
    for i in range(a):
        lat = lis[i][0]
        lng = lis[i][1]
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    x = x / area
    y = y / area

    return [int(x), int(y)]
def get_single_centerpoint(mask):
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one

    count = contour[0][:, 0, :]
    try:
        center = get_centerpoint(count)
    except:
        x,y = count.mean(axis=0)
        center=[int(x), int(y)]

        # max_points = 360
        # if len(contour[0]) > max_points:
        #     compress_rate = len(contour[0]) // max_points
        #     contour[0] = contour[0][::compress_rate, ...]
    return center, contour

def get_coordinates(c_x, c_y, pos_mask_contour,number=4):
        step = int(360//number)
        ct = pos_mask_contour[:, 0, :]
        #print(ct)
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        #print(x,y)
        # angle = np.arctan2(x, y)*180/np.pi
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x ** 2 + y ** 2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]

        #生成number个角度
        new_coordinate = {}
        for i in range(0, 360, step):
            if i in angle:
                d = dist[angle==i].max()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i+1].max()
                new_coordinate[i] = d
            elif i - 1 in angle:
                d = dist[angle == i-1].max()
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i+2].max()
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i-2].max()
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i+3].max()
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i-3].max()
                new_coordinate[i] = d


        distances = torch.zeros(number)

        for a in range(0, 360, step):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[a//step] = 1e-6
            else:
                distances[a//step] = new_coordinate[a]
        # for idx in range(36):
        #     dist = new_coordinate[idx * 10]
        #     distances[idx] = dist

        return distances, new_coordinate
