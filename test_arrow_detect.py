from __future__ import print_function

import math

import cv2 as cv
import numpy as np

import random as rng
rng.seed(12345)


def near_color(color, target_color):
  return ((abs(color[0] - target_color[0]) <= 50)
          and (abs(color[1] - target_color[1]) <= 50)
          and (abs(color[2] - target_color[2]) <= 50))

def near_coord(coord, target_cord):
  return ((abs(coord[0] - target_cord[0]) <= 5)
          and (abs(coord[1] - target_cord[1]) <= 5))

def near_num(num, target_num):
  return (abs(num - target_num) <= 5)

def point_in_conture(contur, point):
  point = [int(point[0]), int(point[1])]
  res = cv.pointPolygonTest(contur, point, False)
  if res > 0:
    return point
  point[0] += 1
  res = cv.pointPolygonTest(contur, point, False)
  if res > 0:
    return point
  point[0] -= 2
  res = cv.pointPolygonTest(contur, point, False)
  if res > 0:
    return point
  point[0] += 1
  point[1] += 1
  res = cv.pointPolygonTest(contur, point, False)
  if res > 0:
    return point
  point[1] -= 2
  res = cv.pointPolygonTest(contur, point, False)
  if res > 0:
    return point
  return None

def thresh_callback(val):
  threshold = val
 # Detect edges using Canny
  canny_output = cv.Canny(src_gray, threshold, threshold * 2)
 # Find contours
  contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
 # Find the convex hull object for each contour
  hull_list = []
  defects_list = []
  contours_poly = [None] * len(contours)
  centers = [None] * len(contours)
  radius = [None] * len(contours)
  drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
  for i in range(len(contours)):
    contours_poly_i = cv.approxPolyDP(contours[i], 3, True)
    centers_i, radius_i = cv.minEnclosingCircle(contours_poly_i)
    flag = False
    for z in range(i):
      if centers[z] is not None and (near_coord(centers[z], centers_i) and near_num(radius[z], radius_i)):
        flag = True
    if flag: continue
    contours_poly[i] = contours_poly_i
    centers[i] = centers_i
    radius[i] = radius_i
    hull = cv.convexHull(contours[i], returnPoints = False)
    try:
      defects = cv.convexityDefects(contours[i], hull)
    except Exception:
      continue
    hull_list.append(hull)
    defects_list.append(defects)
    if defects is not None:
      for j in range(defects.shape[0]):
        s, e, f, d = defects[j, 0]
        start = tuple(contours[i][s][0])
        end = tuple(contours[i][e][0])
        far = tuple(contours[i][f][0])
        line_center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        dist = math.sqrt(math.pow(line_center[0] - centers[i][0], 2) + math.pow(line_center[1] - centers[i][1], 2))
        point = point_in_conture(contours[i], far)
        if not point: continue
        color = src[point[1], point[0]]  # (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        color = (int(color[0]), int(color[1]), int(color[2]))
        if dist < (radius[i] * 0.35) and near_color(color, (0, 145, 220)):
          cv.line(drawing, start, end, [0, 255, 0], 2)
          cv.circle(drawing, line_center, 5, [0, 255, 255], -1)
          cv.circle(drawing, far, 5, [0, 0, 255], -1)
          cv.drawContours(drawing, contours, i, color)
          # cv.drawContours(drawing, hull_list, i, color)

          cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)
          cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i] * 0.35), color, 1)

    # Show in a window
  cv.imshow('Contours', drawing)
# Load source image
src = cv.imread('arrow.png', cv.IMREAD_COLOR)
if src is None:
  print('Could not open image')
  exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()