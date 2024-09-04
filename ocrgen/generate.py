import os
import random
import string
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.random import default_rng

from .augment import transform_image

def get_OCR_data(content, canvas_resolution, text_area, font_file_path):
    w = text_area[0]
    h = text_area[1]
    img = Image.new('RGB', canvas_resolution, color = (0, 0, 0))
    fnt = ImageFont.truetype(font_file_path, 200)
    draw = ImageDraw.Draw(img)
    draw.text((512,512), content, font=fnt, fill=(255, 255, 255))


    img_original = np.uint8(np.copy(img))
    img, transforms = transform_image(np.uint8(img_original),20,10,5,brightness=0)
    Rot_M, Trans_M, Shear_M = transforms
    rot = np.vstack([Rot_M,[0,0,1]])
    trans = np.vstack([Trans_M,[0,0,1]])
    shear = np.vstack([Shear_M,[0,0,1]])

    BBs = []
    MASKs = []
    img_temp = Image.new('RGB', canvas_resolution, color = (0, 0, 0))
    draw_temp = ImageDraw.Draw(img_temp)
    mask = np.array(img_temp)>0
    for letter_idx in range(len(content)):
        if content[letter_idx]==' ':
            continue
        partial = content[:letter_idx+1]
        draw_temp.text((512,512), partial, font=fnt, fill=(255, 255, 255))
        img_temp_numpy = np.array(img_temp)
        img_temp_numpy_masked = img_temp_numpy*(1-np.uint8(mask))
        Xs = np.argwhere(img_temp_numpy_masked>0)[:,1]
        Ys = np.argwhere(img_temp_numpy_masked>0)[:,0]
        border = 2
        x = Xs.min() - border
        w = Xs.max() - Xs.min() + border
        y = Ys.min() - border
        h = Ys.max() - Ys.min() + border
        img_rect = cv2.rectangle(np.copy(img_temp_numpy), (x, y), (x + w, y + h), (0,255,0), border)
        mask = img_temp_numpy>0
        BBs.append([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        MASKs.append(mask)

    BBs_transformed = []
    for box in BBs:
        # transform bounding boxes
        box = np.float32(box)
        box_t = cv2.perspectiveTransform(np.array([box]), rot)
        box_t = cv2.perspectiveTransform(box_t, trans)
        box_t = cv2.perspectiveTransform(box_t, shear)
        BBs_transformed.append(box_t)


    # estimate cropping window
    Xs = np.argwhere(img>0)[:,1]
    Ys = np.argwhere(img>0)[:,0]
    pad = 100
    x = Xs.min() - pad
    w = Xs.max() - x + pad
    y = Ys.min() - pad
    h = Ys.max() - y + pad

    # crop image
    img = img[y:y+h,x:x+w, :]
    img_bb = np.copy(img)
    # adjust bounding boxes for cropping
    for box_t in BBs_transformed:
        box_t[:,:,0]  = box_t[:,:,0] - x
        box_t[:,:,1]  = box_t[:,:,1] - y
        img_bb = cv2.drawContours(img_bb, np.int32(box_t), 0, (128, 255, 128), 2)

    # Mask
    rng = default_rng()
    red = rng.choice(255, size=128, replace=False)
    green = rng.choice(255, size=128, replace=False)
    blue = rng.choice(255, size=128, replace=False)
    img_seg = np.copy(img)
    img_annot = np.copy(img_bb)
    mask_prev = np.zeros(MASKs[0].shape, np.bool)
    for idx in range(len(MASKs)):
        mask = MASKs[idx]
        mask[mask_prev] = False
        mask_prev = mask_prev | mask
        mask = np.uint8(mask)*255
        mask = cv2.warpAffine(mask,Rot_M,canvas_resolution)
        mask = cv2.warpAffine(mask,Trans_M,canvas_resolution)
        mask = cv2.warpAffine(mask,Shear_M,canvas_resolution)
        mask = mask[y:y+h,x:x+w, :]
        color = np.zeros_like(img_seg)
        color[:,:,0] = blue[ord(content[idx])%256]
        color[:,:,1] = green[ord(content[idx])%256]
        color[:,:,2] = red[ord(content[idx])%256]
        color = np.uint8(color)
        img_seg[mask>0] = color[mask>0]
        img_annot[mask>0] = color[mask>0]
    
    return img, img_seg, img_bb, img_annot

