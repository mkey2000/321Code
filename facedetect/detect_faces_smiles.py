import cv2
import sys

imagePath = sys.argv[1]
faceCascPath = "classifier/haarcascade_frontalface_default.xml"
smileCascPath = "classifier/haarcascade_smile.xml"
green = (0, 255, 0)
blue = (255, 0, 0)

faceCascade = cv2.CascadeClassifier(faceCascPath)
smileCascade = cv2.CascadeClassifier(smileCascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=15,
    minSize=(30, 30),
)

print(f'Found {len(faces)} faces!')
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), green, 2)
    face = image[y:y+h, x:x+w]
    face_gray = gray[y:y+h, x:x+w]

    smiles = smileCascade.detectMultiScale(
        face_gray,
        scaleFactor=1.5,
        minNeighbors=13,
        minSize=(15, 15)
    )

    for (xs, ys, ws, hs) in smiles:
        cv2.rectangle(face, (xs, ys), (xs+ws, ys+hs), blue, 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
