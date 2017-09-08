import cv2

cap = cv2.VideoCapture(1)

faceCascade = cv2.CascadeClassifier("classifier/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("classifier/haarcascade_smile.xml")

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        smiles = smileCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=6,
            minSize=(30,30),
        )
        print(f'Found {len(smiles)} smiles!')
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_color,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
