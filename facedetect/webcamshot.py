import cv2
import sys

dev = sys.argv[1] if len(sys.argv) >= 2 else 0
cap = cv2.VideoCapture(int(dev))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);

def save_image(image):
    filename = f'output/webcamshot.jpg'
    print(f'taking picture: {filename}')
    cv2.imwrite(filename, image)

while True:
    ret, frame = cap.read()
    cv2.imshow('321Code!', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save_image(frame)
        break

cap.release()
cv2.destroyAllWindows()
