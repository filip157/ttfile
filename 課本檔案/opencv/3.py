import cv2

cap = cv2.VideoCapture("movie.mp4")


if cap.isOpened():
	while(1):
		ret, frame = cap.read()
		cv2.imshow("1",frame)
		if cv2.waitKey(2) & 0xff == ord('q'):
			break
cap.release()
cv2.destroyAllWindows()

img = cv2.imread('dice.png')