import cv2
import numpy as np

# This is a special version of the _tools/process.py script
# It has different parameters because first scan wasn't directly from the scanner
# and had an applied gamma, etc.

def adjust_gamma(image, gamma=1.0, dtype='float32'):
  invGamma = 1.0 / gamma
  table = np.array([ (i / 255.0) ** invGamma for i in range(256) ], dtype=dtype)
  return cv2.LUT(image, table)
weights = np.array((0.0722, 0.7152, 0.2126), dtype='float32')
get_luminance = lambda orig: np.inner(adjust_gamma(orig, 1/2.2), weights) # orig should be BGR!


approx_pitch = 59
radius = 3
b_radius = 1.5

coords = [
  ( "page1.png",
    (23.5, 22),
    (14.5, 2318.5),
    (1432, 2323) ),

  ( "page2.png",
    (24.5, 39.5),
    (20, 2339),
    (1556.5, 2342.5) ),
]

for centry in coords:
  # calculate grid base
  p1, p2, p3 = ( np.array(x) for x in centry[1:] )
  dy = (p2 - p1)
  ny = int(round(np.linalg.norm(dy) / approx_pitch))
  dx = (p3 - p2)
  nx = int(round(np.linalg.norm(dx) / approx_pitch))
  offset = p1
  base = np.array((dx/nx, dy/ny)).T
  to_base = lambda p: offset + np.inner(base, p)

  # load and preprocessing
  filename = centry[0]
  print('Processing', filename)
  im = cv2.imread(filename)
  im_y = get_luminance(im)

  # extract dots
  k1 = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3,3))
  dots = cv2.morphologyEx((im_y < .0342).astype('uint8') * 255, cv2.MORPH_DILATE, k1, iterations=2).astype('bool')

  # paint grid
  #for x in range(nx + 1):
  #  for y in range(ny + 1):
  #    pt = np.round(to_base((x, y)) - 0.5).astype(int)
  #    cv2.circle(im, tuple(pt), radius, (255,255,255), -1)

  # patch dots
  im_b = cv2.GaussianBlur(im_y * ~dots, (0, 0), b_radius) / cv2.GaussianBlur((~dots).astype('float32'), (0, 0), b_radius)
  im_p = (im_y * ~dots) + im_b * dots

  # final processing and save
  fi = cv2.morphologyEx((im_p < 183/255).astype('uint8') * 255, cv2.MORPH_CLOSE, k1, iterations=1)
  cv2.imwrite('proc-' + filename, np.round(255 - fi).astype('uint8'))
