import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = ("C:/Users/extnoriegasl/OneDrive - Balluff GmbH/Documentos/EstadiasLCSN/tulipanes.jpg")
image = cv2.imread(image_path)
if image is None:
    exit()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
colors = ('b', 'g', 'r')
channels = cv2.split(image)
plt.figure(figsize=(10, 5))
plt.title("Histogramas de los canales de color")
plt.xlabel("Intensidad")
plt.ylabel("Frecuencia")
for i, color in enumerate(colors):
    hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f"Canal {color.upper()}")
    plt.xlim([0, 256])
plt.legend()
plt.show()