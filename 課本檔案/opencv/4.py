#導入open-cv2
import cv2

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