print('\n***** face recognition using open cv library - Mauricio Ribeiro *****\n')

import cv2, os

cascPath = os.getcwd() + '/opencv-3.0.0/data/haarcascade/'
faceXml = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath + faceXml)

webcam  = cv2.VideoCapture(0)

while True:
	ret, frame = webcam.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	scaleFactor = 1.3
	minNeighbors = 5

	faces = faceCascade.detectMultiScale(gray, scaleFactor, minNeighbors)
	color = (0, 255, 0)
	thickness = 2

	for (x, y, width, height) in faces:
		cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)

	cv2.imshow('Face tracking record', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

webcam.release()
cv2.destroyAllWindows()
