import cv2
import numpy as np

# resize image with original ratio
def resizeImage(img, size):
    jpg = img
    shape = jpg.shape[:2]
    r = min(size[0] / shape[0], size[1] / shape[1])
    new_size = int(round(shape[0] * r)), int(round(shape[1] * r))
    border = int((size[0] - new_size[0]) / 2), int((size[1] - new_size[1]) / 2)
    jpg = cv2.resize(jpg, (new_size[1], new_size[0]))
    num = np.zeros((size[0], size[1], 3), np.uint8) + 255
    num[border[0]:new_size[0]+border[0], border[1]:new_size[1]+border[1]] = jpg 
    return num

# crop image function
def cropImage(image, c1, c2):
    num = np.zeros((c2[0]-c1[0], c2[1]-c1[1], 3), np.uint8)
    num = image[c1[0]:c2[0], c1[1]:c2[1]]
    return num

# scale reactangle function
def scaleRectangle(shape, c1, c2, factor = 1):
    squareImage = shape[0] * shape[1]
    squareRectangle = (c2[0] - c1[0]) * (c2[1] - c1[1])
    gain = (squareRectangle / squareImage)**factor
    dx = int(c1[0] * gain)
    dy = int(c1[1] * gain)
    do = min(dx, dy)
    c1 = c1[0] - do, c1[1] - do
    c2 = c2[0] + do, c2[1] + do
    return c1, c2
