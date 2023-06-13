import cv2
import numpy as np

img = cv2.imread('dice.png')
height, width = img.shape[:2]
print('圖片像素高度: ', height)
print('圖片像素寬度: ', width)

quarter_height, quarter_width = height / 4, width / 4
print('垂直移動像素: ', quarter_height)
print('水平移動像素: ', quarter_width)

n = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

print('移動矩陣:\n', n)
img_translation = cv2.warpAffine(img, n, (width, height))
cv2.imshow('Before translation',img)
cv2.imshow('After translation',img_translation)
cv2.waitKey(0)
cv2.destroyAllWindows()