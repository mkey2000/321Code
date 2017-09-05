import cv2
import sys

imagePath = sys.argv[1]
faceCascPath = "classifier/haarcascade_frontalface_default.xml"
green = (0, 255, 0)

faceCascade = cv2.CascadeClassifier(faceCascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

print(f'Found {len(faces)} faces!')
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), green, 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
