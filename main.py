import os
import random
import string
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.random import default_rng

from ocrgen.generate import get_OCR_data

word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.content.splitlines()
word = random.choice(WORDS)
if random.randint(0,1)==0: word = word.upper()
text = word.decode()
text = "HELLO!"

font_files_root = '/ssd_data/common/OCRchestra/text_generator/font_library/Microsoft-365-Fonts/'
# font_file_path = f"/ssd_data/common/OCRchestra/text_generator/font_library/Others/mangal.ttf"
#font_file_path = f"/ssd_data/common/OCRchestra/text_generator/font_library/Others/suravaram.ttf"

#TEXTS = ["MEDICAL","Total","PHONE","please","YOLO!","Where?","'QUOTE'"]
# TEXTS = ["भूगोल","सड़क","लगभग"]
#TEXTS = ["చమక","బహ","అంధ"]
TEXTS = ["RANDOM PHRASE!","Random Phrase!","random phrase!"]

for idx in range(len(TEXTS)):
    text = TEXTS[idx]
    subdir = random.choice(os.listdir(font_files_root))
    font_files = os.listdir(os.path.join(font_files_root,subdir))
    font_file_path = f"{font_files_root}/{subdir}/{random.choice(font_files)}"

    img, img_seg, img_rect, img_annot = get_OCR_data(content=text, canvas_resolution=(2048,2048), text_area=(1024,1024), font_file_path = font_file_path)
    cv2.imwrite(f"out/{str(idx)}_text.jpg", img)
    cv2.imwrite(f"out/{str(idx)}_seg.jpg", img_seg)
    cv2.imwrite(f"out/{str(idx)}_boxes.jpg", img_rect)
    cv2.imwrite(f"out/{str(idx)}_annot.jpg", img_annot)
