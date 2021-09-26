import cv2
import numpy as np

img = np.zeros(shape=(300, 300, 3), dtype=np.int8)
vertices = np.array(
    [
     [0, 0],
     [300, 0],
     [0, 300]
    ], dtype=np.int32
)

vertices = vertices.reshape((-1, 1, 2))
cv2.fillPoly(
    img,
    [vertices],
    color=[255, 255, 255]
)
while True:
    cv2.imshow('Obraz', img)

    # do stuff

    if cv2.waitKey(20) & 0xff == 27:
        break
