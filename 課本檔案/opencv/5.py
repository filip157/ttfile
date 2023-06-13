#導入open-cv2
import cv2

img = cv2.imread('dice.png')
cv2.imshow('origianl',img)
cv2.waitKey(0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayscaleImage",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()