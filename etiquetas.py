import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
 
# Definir modelo de red neuronal
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # 2 clases: "Correcta" o "Incorrecta"
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# Crear y cargar modelo (sustituir por un modelo entrenado si ya tienes uno)
model = create_model()

# Etiquetas: 0 = Incorrecta, 1 = Correcta
labels = ["Incorrecta", "Correcta"]
 
# Captura en tiempo real
cap = cv2.VideoCapture(0)  # Usa la cámara predeterminada
print("Presiona 'q' para salir.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break
 
    # Preprocesamiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=-1)  # Añadir canal
    input_image = np.expand_dims(input_image, axis=0)  # Añadir dimensión batch
 
    # Predicción
    predictions = model.predict(input_image)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]
 
    # Mostrar resultados
    text = f"{labels[class_id]} ({confidence:.2f})"
    color = (0, 255, 0) if class_id == 1 else (0, 0, 255)
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
 
    cv2.imshow('Sistema de Visión en Tiempo Real', frame)
 
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Liberar recursos
cap.release()
cv2.destroyAllWindows()

#BCC M425'0000'1A'003'EX44T2'050
#BCC0577BCCM425-0000-1A-06-PX0334-100
