import cv2
import numpy as np

img = cv2.imread('dice.png')
height, width = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, .5)

rotataed_image = cv2.warpAffine(img, rotation_matrix, (width, height))
cv2.imshow('Rotated Image', rotataed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()