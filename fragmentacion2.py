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

    # Create the mask for the foreground and background
    mask_new = np.where((seg == 2) | (seg == 0), 0, 1).astype("uint8")
    img_result = img * mask_new[:, :, np.newaxis]  # Apply mask
    cv2.imshow("Output", img_result)

def box_draw(click, x, y, flags, param):
    global x_pt, y_pt, drawing, topleft_pt, bottomright_pt, img, image

    if click == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_pt, y_pt = x, y

    elif click == cv2.EVENT_MOUSEMOVE:
        if drawing:
            topleft_pt, bottomright_pt = (x_pt, y_pt), (x, y)
            # Draw a rectangle while dragging the mouse
            temp_img = img.copy()
            cv2.rectangle(temp_img, topleft_pt, bottomright_pt, (0, 255, 0), 2)
            cv2.imshow("Frame", temp_img)

    elif click == cv2.EVENT_LBUTTONUP:
        drawing = False
        topleft_pt, bottomright_pt = (x_pt, y_pt), (x, y)
        cv2.rectangle(image, topleft_pt, bottomright_pt, (0, 255, 0), 2)
        bounding_box = (x_pt, y_pt, x - x_pt, y - y_pt)

        # Call the GrabCut function
        algo_grabcut(img, bounding_box)

drawing = False
topleft_pt, bottomright_pt = (-1, -1), (-1, -1)

img = cv2.imread("C:/Users/extnoriegasl/OneDrive - Balluff GmbH/Documentos/EstadiasLCSN/tulipanes.jpg")
if img is None:
    print("Error: Image not found.")
    exit()

img = cv2.resize(img, (500, 500))
image = img.copy()
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", box_draw)

while True:
    cv2.imshow("Frame", image)
    ch = cv2.waitKey(1)
    if ch == 32:  # Press space to exit
        break

cv2.destroyAllWindows()
