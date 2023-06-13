import numpy as np
import cv2

img = cv2.imread("dice.png")
cv2.imshow('Title',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("dice.png")
cv2.imshow('Title',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('dice.png')
cv2.imwrite('output.jpg',img)
cv2.imread('output.jpg')
cv2.imshow('copy_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap = cv2.VideoCapture("movie.mp4")


if cap.isOpened():
	while(1):
		ret, frame = cap.read()
		cv2.imshow("1",frame)
		if cv2.waitKey(2) & 0xFF == ord('q'):
			break
cap.release()
cv2.destroyAllWindows()

img = cv2.imread('dice.png')

#將圖片讀入img
img = cv2.imread('dice.png')
#顯示img，開啟名稱為copy_img
cv2.imshow('copy_img',img)
#輸出img的shape
print(img.shape)
print("高度的像素有: ", img.shape[0])
print("寬度的像素有: ", img.shape[1])
#open-cv2的等待
cv2.waitKey(0)
#open-cv2的關閉所有介面(由cv2開啟的)
cv2.destroyAllWindows()

img = cv2.imread('dice.png')
cv2.imshow('origianl',img)
cv2.waitKey(0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayscaleImage",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('dice.png')
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image',img_HSV)
cv2.imshow('Hue Channel',img_HSV[:, :,0])
cv2.imshow('Saturation Channel',img_HSV[:, :,1])
cv2.imshow('Value Channel',img_HSV[:, :,2])
cv2.waitKey(0)
cv2.destroyAllWindows()

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

img = cv2.imread('dice.png')
height, width = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, .5)

rotataed_image = cv2.warpAffine(img, rotation_matrix, (width, height))
cv2.imshow('Rotated Image', rotataed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()