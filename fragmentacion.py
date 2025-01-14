import cv2
import numpy as np

image = cv2.imread('C:/Users/extnoriegasl/OneDrive - Balluff GmbH/Documentos/EstadiasLCSN/tulipanes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply thresholding
ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#eliminar ruido
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)

# Find background region
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Find foreground region
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Create marker image
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Apply Watershed algorithm
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
cv2.watershed(image, markers)

# Apply colormap to the markers
colored_markers = np.zeros_like(image)
colored_markers[markers == -1] = [183, 247, 239]  #color de fradmento

# Display the segmented image
segmented_image = cv2.addWeighted(image, 0.7, colored_markers, 0.3, 0)
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)