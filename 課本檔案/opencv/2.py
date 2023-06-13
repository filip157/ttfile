import cv2

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