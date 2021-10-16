import cv2, os

cascPath = os.getcwd()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

webcam = cv2.VideoCapture(0)

while True:
	ret, frame = webcam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	scaleFactor = 1.3
	minNeighbors = 5
	thickness = 2
	faceColor, eyeColor = (0, 255, 0), (255, 0, 0)

	faces = faceCascade.detectMultiScale(gray, scaleFactor, minNeighbors)
	for (fx, fy, fwidth, fheight) in faces:
		cv2.rectangle(frame, (fx, fy), (fx + fwidth, fy + fheight), faceColor, thickness)

		faceGray = gray[fy:fy + fheight, fx:fx + fwidth]
		eyes = eyeCascade.detectMultiScale(faceGray)
		if len(eyes) <= 2:
			for (ex, ey, ewidth, eheight) in eyes:
				cv2.rectangle(frame[fy:fy + fheight, fx:fx + fwidth], (ex, ey), (ex + ewidth, ey + eheight), eyeColor,
							  thickness)

	cv2.imshow('Face tracking record', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

webcam.release()
cv2.destroyAllWindows()
