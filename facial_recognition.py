
import cv2
from deepface import DeepFace
import threading
import os
from queue import Queue

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_match = False
counter = 0
frame_queue = Queue()

reference_img = cv2.imread("faces/deepanshu.jpg")
person_name = "Deepanshu"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def check_face():
    global face_match
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]
                    
                    face_roi_resized = cv2.resize(face_roi, (160, 160))

                    if DeepFace.verify(face_roi_resized, reference_img.copy(), model_name='Facenet')['verified']:
                        face_match = True
                    else:
                        face_match = False
            except ValueError:
                face_match = False

threading.Thread(target=check_face, daemon=True).start()

while True:
    ret, frame = cap.read()

    if ret:
        frame_resized = cv2.resize(frame, (320, 240))

        if counter % 5 == 0:
            frame_queue.put(frame_resized)

        counter += 1

        if face_match:
            cv2.putText(frame, f"FACE MATCHED: {person_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "FACE NOT MATCHED!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
