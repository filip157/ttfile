import cv2

img = cv2.imread('dice.png')
cv2.imdecode('origianl',img)
cv2.waitKey(0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
cv2.waitKey(0)
cv2.destroyAllWindows()