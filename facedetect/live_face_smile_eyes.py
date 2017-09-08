import cv2

cap = cv2.VideoCapture(1)

faceCascade = cv2.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("classifier/haarcascade_smile.xml")
eyeCascade = cv2.CascadeClassifier("classifier/haarcascade_eye_tree_eyeglasses.xml")

def label(image, text, x, y, w, h, color):
    cv2.putText(image, text, (x, y-5), 1, .7, color)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)


while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    print("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        label(frame, 'face', x, y, w, h, (255, 0, 0))
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        smiles = smileCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.8,
            minNeighbors=6,
            minSize=(30,30),
        )
        print(f'Found {len(smiles)} smiles!')
        for (sx, sy, sw, sh) in smiles:
            label(face_color, 'smile', sx, sy, sw, sh, (0, 0, 255))

        eyes = eyeCascade.detectMultiScale(
            face_gray
        )
        print(f'Found {len(eyes)} eyes!')
        for (ex, ey, ew, eh) in eyes:
            label(face_color, 'eyes', ex, ey, ew, eh, (0, 255, 0))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
