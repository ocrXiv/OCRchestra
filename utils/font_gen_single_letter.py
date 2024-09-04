import os
import random
import string
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from text_generator.ocrgen.augment import transform_image


word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.content.splitlines()
word = random.choice(WORDS)
if random.randint(0,1)==0: word = word.upper()
content = word.decode()
content = "A"[:1]

font_files_root = '/ssd_data/common/OCRchestra/text_generator/font_library/Microsoft-365-Fonts/'

subdir = random.choice(os.listdir(font_files_root))
font_files = os.listdir(os.path.join(font_files_root,subdir))
font_file_path = f"{font_files_root}/{subdir}/{random.choice(font_files)}"

# font_file_path = f"/ssd_data/common/OCRchestra/text_generator/font_library/Others/mangal.ttf"

w = random.randint(1024,1024)
h = random.randint(1024,1024)
img = Image.new('RGB', (2048, 2048), color = (0, 0, 0))
fnt = ImageFont.truetype(font_file_path, 150)
d = ImageDraw.Draw(img)
d.text((512,512), content, font=fnt, fill=(255, 255, 255))

# img = cv2.rectangle(img, (x-border, y-border), (x + w + border, y + h + border), (0,255,0), border)

# img = img[y:y+h,x:x+w, :]

img_original = np.uint8(np.copy(img))
img, transforms = transform_image(np.uint8(img_original),20,10,5,brightness=0)
Rot_M, Trans_M, Shear_M = transforms
rot = np.vstack([Rot_M,[0,0,1]])
trans = np.vstack([Trans_M,[0,0,1]])
shear = np.vstack([Shear_M,[0,0,1]])
contours, hierarchy = cv2.findContours(img_original[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(img[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total letters:{len(content)} Total BBs:{len(contours)}")
for contour in contours:
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # transform bounding boxes
    box_t = cv2.perspectiveTransform(np.array([box]), rot)
    box_t = cv2.perspectiveTransform(box_t, trans)
    box_t = cv2.perspectiveTransform(box_t, shear)
    box_t = np.int32(box_t)

# crop window
Xs = np.argwhere(img>0)[:,1]
Ys = np.argwhere(img>0)[:,0]
pad = 100
x = Xs.min() - pad
w = Xs.max() - x + pad
y = Ys.min() - pad
h = Ys.max() - y + pad

# crop image
img = img[y:y+h,x:x+w, :]

# adjust bounding box for cropping
box_t[:,:,0]  = box_t[:,:,0] - x
box_t[:,:,1]  = box_t[:,:,1] - y
cv2.drawContours(img, box_t, 0, (0, 255, 0), 2)

cv2.imwrite('pil_text_font.png',img)