import cv2

img = cv2.imread('C:\\Users\\admin\\Desktop\\dice.png')
cv2.imshow('copy_img',img)
print(img.shape)
print("高度的像素有: ", img.shape[0])
print("寬度的像素有: ", img.shape[1])
cv2.waitKey(0)
cv2.destroyAllWindows()