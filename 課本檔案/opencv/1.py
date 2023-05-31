import cv2

img = cv2.imread("C:\\Users\\admin\\Desktop\\dice.png")
cv2.imwrite('C:\\Users\\admin\\Desktop\\Title',img)
cv2.imshow('Title')
cv2.waitKey(0)
cv2.destroyAllWindows()


