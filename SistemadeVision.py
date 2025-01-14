import cv2
import numpy as np
import tensorflow as tf
import time
 
# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_soldadura.h5')
 
# Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))  # Tamaño para el modelo
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0, -1))
 
# Inicializar la cámara
cap = cv2.VideoCapture(0)
 
while True:
    inicio = time.time()  # Temporizador
 
    # Captura de imagen
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara")
        break
 
    # Dividir la imagen en segmentos
    alto, ancho, _ = frame.shape
    segmentos = 4  # Número de segmentos
    ancho_segmento = ancho // segmentos
 
    for i in range(segmentos):
        segmento = frame[:, i * ancho_segmento:(i + 1) * ancho_segmento]
        
        # Preprocesar y predecir
        segmento_preprocesado = preprocesar_imagen(segmento)
        prediccion = model.predict(segmento_preprocesado, verbose=0)[0][0]
 
        # Mostrar resultados
        if prediccion > 0.5:  # Umbral de decisión
            estado = "DEFECTO DETECTADO"
            color = (0, 0, 255)  # Rojo
        else:
            estado = "Soldadura Correcta"
            color = (0, 255, 0)  # Verde
 
        # Mostrar resultados en pantalla
        cv2.rectangle(frame,
                      (i * ancho_segmento, 0),
                      ((i + 1) * ancho_segmento, alto),
                      color, 2)
        cv2.putText(frame, estado,
                    (i * ancho_segmento + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
 
    fin = time.time()
    tiempo_segmento = fin - inicio
 
    # Verifica si cumple con el tiempo máximo permitido
    if tiempo_segmento > 1:
        print(f"Segmento procesado en {tiempo_segmento:.2f} segundos")
 
    cv2.imshow('Análisis de Soldadura', frame)
 
    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()