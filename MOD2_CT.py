#!/usr/bin/env python
# coding: utf-8

import cv2
import urllib.request
import numpy as np
from pathlib import Path


# if dog.jpg doesn't exist in folder, this will download from URL
my_file = Path('dog.jpg')

if my_file.is_file():
    img = cv2.imread('dog.jpg')
else:
    req = urllib.request.urlopen('https://drive.usercontent.google.com/download?id=1G8mj1hakIsifG4aFKtSGLD5p8BdnRujv')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    cv2.imwrite('dog.jpg',img)
    


# show original then split to RGB
cv2.imshow('Original',img)
cv2.waitKey(0)

b,g,r = cv2.split(img)
cv2.imshow('Red', r)
cv2.imshow('Green', g)
cv2.imshow('Blue', b)

cv2.waitKey(0)
cv2.destroyAllWindows()

# merges BGR in that order
image_merge = cv2.merge([b,g,r]) 
cv2.imshow("RGB Merge", image_merge) 
  
cv2.waitKey(0) 
cv2.destroyAllWindows()


# swaps red and green channel and merges
rg_swap= cv2.merge([b,r,g]) 
cv2.imshow("Red and Green swap", rg_swap) 
 
cv2.waitKey(0) 
cv2.destroyAllWindows()





