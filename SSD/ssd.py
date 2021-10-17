import numpy as np
import cv2

# https://github.com/chuanqi305/MobileNet-SSD/blob/master/voc/MobileNetSSD_deploy.prototxt
PROTOTXT = "MobileNetSSD_deploy.prototxt"

# https://github.com/C-Aniruddh/realtime_object_recognition/blob/master/MobileNetSSD_deploy.caffemodel
MODEL = "MobileNetSSD_deploy.caffemodel"

# https://pixabay.com/videos
# https://pixabay.com/videos/cars-motorway-speed-motion-traffic-1900/
INP_VIDEO_PATH = 'horses.mp4'
OUT_VIDEO_PATH = 'detection.mp4'

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Załadowujemy model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Wczytujemy wideo
cap = cv2.VideoCapture(INP_VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter(
    OUT_VIDEO_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    size
)

while True:
    ret, frame = cap.read()
    if not ret:
       break
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
       confidence = detections[0, 0, i, 2]
       if confidence > 0.5:
           idx = int(detections[0, 0, i, 1])
           box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
           (startX, startY, endX, endY) = box.astype("int")
           label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
           cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
           y = startY - 15 if startY - 15 > 15 else startY + 15
           cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[idx], 3)

    result.write(frame)
    # cv2.imshow('detection', frame)
