import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
 
# Cargar el modelo entrenado
modelo = load_model('modelo_soldadura_cnn.h5')
 
# Función para preprocesar las imágenes antes de la predicción
def preprocesar_imagen(imagen):
    imagen = cv2.resize(imagen, (224, 224))  # Redimensionar
    imagen = imagen / 255.0  # Normalizar
    return np.expand_dims(imagen, axis=0)  # Añadir dimensión extra para el batch
 
# Inicializar la cámara
cap = cv2.VideoCapture(0)
 
print("Iniciando el sistema de visión en tiempo real...")
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara")
        break
 
    # Dividir la imagen en segmentos
    alto, ancho, _ = frame.shape
    segmentos = 4
    ancho_segmento = ancho // segmentos
 
    for i in range(segmentos):
        segmento = frame[:, i * ancho_segmento:(i + 1) * ancho_segmento]
        
        # Preprocesar el segmento
        imagen = preprocesar_imagen(segmento)
        
        # Predicción con el modelo
        prediccion = modelo.predict(imagen)[0][0]
        
        # Mostrar resultados
        if prediccion > 0.5:
            estado = "DEFECTO DETECTADO"
            color = (0, 0, 255)  # Rojo
        else:
            estado = "Soldadura Correcta"
            color = (0, 255, 0)  # Verde
 
        cv2.rectangle(frame,
                      (i * ancho_segmento, 0),
                      ((i + 1) * ancho_segmento, alto),
                      color, 2)
        cv2.putText(frame, estado,
                    (i * ancho_segmento + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
 
    cv2.imshow('Análisis de Soldadura y puente', frame)
 
    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()