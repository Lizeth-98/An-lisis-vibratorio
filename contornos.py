import cv2
import numpy as np

#caragar la imagen
imagen = cv2.imread("C:/Users/extnoriegasl/OneDrive - Balluff GmbH/Documentos/EstadiasLCSN/tulipanes.jpg")
#convertir la escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

#APLIAR DESENFOQUE GAUSSIANO/ FOLTRADO
gris_desfocada = cv2.GaussianBlur(gris, (5,5), 0)

#deteccion de bordes con canny /contornos
contornos = cv2.Canny(gris_desfocada, 100, 200)

#mostrar imagen original 
cv2.imshow("original", imagen)

#,ostrar la imagen de contornos
cv2.imshow("contornos", contornos)

#mostrar imagen filtrada
cv2.imshow("Imagen filtrada", gris_desfocada)

#presionar tecla para cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()