import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.processing.image import ImageDataGenerator
import cv2
import numpy as np
import time
 
# Parámetros de entrenamiento
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
 
# Directorios de datos
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
 
# Generadores de datos
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=20,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)
 
val_gen = ImageDataGenerator(rescale=1./255)
 
# Carga de datos
train_data = train_gen.flow_from_directory(train_dir, 
                                           target_size=IMG_SIZE, 
                                           batch_size=BATCH_SIZE, 
                                           class_mode='binary')
 
val_data = val_gen.flow_from_directory(val_dir, 
                                       target_size=IMG_SIZE, 
                                       batch_size=BATCH_SIZE, 
                                       class_mode='binary')
 
# Creación del modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
 
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
 
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
 
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Clasificación binaria
])
 
# Compilación del modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
 
# Entrenamiento del modelo
print("Entrenando el modelo...")
history = model.fit(train_data, 
                    epochs=EPOCHS, 
                    validation_data=val_data)
 
# Guardar el modelo entrenado
model.save('modelo_soldadura.h5')
 
# Evaluación del modelo
loss, accuracy = model.evaluate(val_data)
print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")
 
# Cargar el modelo entrenado para el sistema de visión
model = tf.keras.models.load_model('modelo_soldadura.h5')
 
# Función para preprocesar imágenes
def preprocesar_imagen(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)  # Redimensionar
    normalized = resized / 255.0  # Normalizar
    return np.expand_dims(normalized, axis=(0, -1))
 
# Inicializar la cámara
cap = cv2.VideoCapture(0)
 
print("Iniciando sistema de visión en tiempo real...")
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
        #if predicciones del puente, cheking="el puente es correcto y sin errores"
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
        print(f"Advertencia: Segmento procesado en {tiempo_segmento:.2f} segundos")
 
    cv2.imshow('Análisis de Soldadura', frame)
 
    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
 