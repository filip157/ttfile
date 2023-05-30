import cv2

img = cv2.imread('C:\\Users\\admin\\Desktop\\dice.png')
cv2.imwrite('output.png',img)
cv2.imread('output.png')
cv2.imshow('copy_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()