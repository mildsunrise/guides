#!/usr/bin/env python3

import sys
import cv2
import numpy as np

args = sys.argv[1:]
if len(args) != 2:
  print('Usage: process.py <input> <output>', file=sys.stderr)
  exit(2)

def adjust_gamma(image, gamma=1.0, dtype='float32'):
  invGamma = 1.0 / gamma
  table = np.array([ (i / 255.0) ** invGamma for i in range(256) ], dtype=dtype)
  return cv2.LUT(image, table)
weights = np.array((0.0722, 0.7152, 0.2126), dtype='float32')
get_luminance = lambda orig: np.inner(adjust_gamma(orig, 1/2.2), weights) # orig should be BGR!

dot_dilate = 3
dot_threshold = 15/255
final_threshold = 220/255
b_radius = 1.5

# load and preprocessing
im = cv2.imread(args[0])
im_y = get_luminance(im)

# extract dots
k1 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
dots = cv2.morphologyEx((im_y < dot_threshold).astype('uint8') * 255, cv2.MORPH_DILATE, k1, iterations=dot_dilate).astype('bool')

# patch dots
im_b = cv2.GaussianBlur(im_y * ~dots, (0, 0), b_radius) / cv2.GaussianBlur((~dots).astype('float32'), (0, 0), b_radius)
im_p = (im_y * ~dots) + im_b * dots

# final processing and save
fi = cv2.morphologyEx((im_p < final_threshold).astype('uint8') * 255, cv2.MORPH_CLOSE, k1, iterations=1)
cv2.imwrite(args[1], np.round(255 - fi).astype('uint8'))
