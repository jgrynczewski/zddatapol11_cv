# Automatyczne wskazywanie źródła

import cv2
import numpy as np

# Wczytujemy obraz
road = cv2.imread('road_image.jpg')

# Tworzymy kopię obrazu
road_copy = np.copy(road)

# Pusty obraz na który bedziemy nanosili wynik
print(road.shape[:2])
marker_image = np.zeros(road.shape[:2], dtype=np.int32)

segments = np.zeros(road.shape, dtype=np.uint8)

print(marker_image.shape)
print(segments.shape)

# Do markerów wybieramy kolory z palety tab10
from matplotlib import cm

print(cm.tab10(0))  # R, G, B, A
print(cm.tab10(0)[:3])  # R, G, B

print(tuple(np.array(cm.tab10(0)[:3])*255))


def create_rgb(idx):
    return tuple(np.array(cm.tab10(idx)[:3])*255)


colors = []
for i in range(10):
    colors.append(create_rgb(i))

print(colors)

# 3 komponenty algorytmu wododziałowego:
# a. global variables
# b. callback functiona
# c. while True loop

# 1. GLOBAL VARIABLES
n_markers = 10  # 0-9
current_marker = 1  # Indeks na palecie kolorów
marks_updated = False  # flaga czy marker był aktualizoway przez
# algorytm wododziałowy


# b. CALLBACK FUNCTION
def mouse_callback(event, x, y, flags, param):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:  # rysujemy markery na dwóch obrazach
        # Marker na ilustracji przedstawiającej działanie algorytmu
        cv2.circle(marker_image, (x, y), 10, current_marker, -1)

        # Marker na oryginalnym zdjęciu
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)

        marks_updated = True


# WHILE TRUE
cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouse_callback)

while True:
    cv2.imshow('Watershed Segments', segments)
    cv2.imshow('Road Image', road_copy)

    k = cv2.waitKey(1)

    if k == 27:  # ESC
        break

    # c powoduje wyczyszczenie wszystkich kolorów
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.uint8)

    # Aktualizacja kolorów
    elif k > 0 and chr(k).isdigit():  # jeżeli znak, który wprowadzono z klawiatury
        # jest liczbą i jest większy od 0 to zmieniamy kolor piksela zgodnie
        # z wprwowadzoną liczbą (indeksem)
        current_marker = int(chr(k))

    # Aktualizacja markerów
    if marks_updated:  # marekry są aktualizaowane przy każdym kliknięciu w lewym przisk myszki
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)

        segments = np.zeros(road.shape, dtype=np.uint8)

        for color_ind in range(n_markers):
            # kolorowanie segmentów
            segments[marker_image_copy == color_ind] = colors[color_ind]

cv2.destroyAllWindows()
