import cv2
import time
import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);

faceCascade = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('classifier/haarcascade_smile.xml')

red = (0, 0 , 255)
green = (0, 255, 0)
blue = (255, 0, 0)
min_pct_smiles = .5
min_tandem_smiles = 30
tandem_smiles = 0

def take_picture(image):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'output/image_{now}.png'
    print(f'taking picture: {filename}')
    cv2.imwrite(filename, image)
    time.sleep(5)

while(True):
    num_faces = 0
    num_smiles = 0
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    num_faces = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), red, 1)
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        smiles = smileCascade.detectMultiScale(
            face_gray,
            scaleFactor=1.8,
            minNeighbors=7,
            minSize=(7,5),
        )

        if len(smiles) >= 1:
            num_smiles += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), green, 1)

    if num_faces > 0:
        pct_smiles = num_smiles / num_faces
        if pct_smiles >= min_pct_smiles:
            tandem_smiles += 1
        else:
            tandem_smiles = 0

    frame = cv2.resize(frame, (1024, 768))

    text = (f'faces:{num_faces} '
            f'smiles:{num_smiles} '
            f'tandem_smiles:{min_tandem_smiles - tandem_smiles} ')

    cv2.putText(frame, text, (10, 20), 5, 1, blue)
    if tandem_smiles >= min_tandem_smiles:
        take_picture(frame)
        tandem_smiles = 0

    cv2.imshow('321Code!', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
