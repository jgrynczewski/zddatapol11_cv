import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(
            face_img,
            (x, y),
            (x + w, y + h),
            color=(255, 255, 255),
            thickness=10
        )

    return face_img


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read(0)

    # wszystko co robiliśmy na zdjęciu, możemy zrobić na pojedynczej klatce filmu
    frame = detect_face(frame)

    cv2.imshow('Video face Detect', frame)

    k = cv2.waitKey(1)
    if k == 27:  # if press Esc stop recording
        break


cap.release()
cv2.destroyAllWindows()