import cv2
import numpy as np


def algo_grabcut(img, bounding_box):
    seg = np.zeros(img.shape[:2], np.uint8)
    x, y, width, height = bounding_box
    seg[y : y + height, x : x + width] = 1
    background_mdl = np.zeros((1, 65), np.float64)
    foreground_mdl = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img, seg, bounding_box, background_mdl, foreground_mdl, 5, cv2.GC_INIT_WITH_RECT
    )

    mask_new = np.where((seg == 2) | (seg == 0), 0, 1).astype("uint8")
    img = img * mask_new[:, :, np.newaxis]
    cv2.imshow("Output", img)


def box_draw(click, x, y, flag_param, parameters):
    global x_pt, y_pt, drawing, topleft_pt, bottomright_pt, img

    if click == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_pt, y_pt = x, y

    elif click == cv2.EVENT_MOUSEMOVE:
        if drawing:
            topleft_pt, bottomright_pt = (x_pt, y_pt), (x, y)
            image[y_pt:y, x_pt:x] = 255 - img[y_pt:y, x_pt:x]
            cv2.rectangle(image, topleft_pt, bottomright_pt, (0, 255, 0), 2)

    elif click == cv2.EVENT_LBUTTONUP:
        drawing = False
        topleft_pt, bottomright_pt = (x_pt, y_pt), (x, y)
        image[y_pt:y, x_pt:x] = 255 - image[y_pt:y, x_pt:x]
        cv2.rectangle(image, topleft_pt, bottomright_pt, (0, 255, 0), 2)
        bounding_box = (x_pt, y_pt, x - x_pt, y - y_pt)

        algo_grabcut(img, bounding_box)


drawing = False
topleft_pt, bottomright_pt = (-1, -1), (-1, -1)

img = cv2.imread("C:/Users/extnoriegasl/OneDrive - Balluff GmbH/Documentos/EstadiasLCSN/tulipanes.jpg")
img = cv2.resize(img, (500, 500))
image = img.copy()
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", box_draw)

while True:
    cv2.imshow("Frame", image)
    ch = cv2.waitKey(1)
    if ch == 32:
        break

cv2.destroyAllWindows()