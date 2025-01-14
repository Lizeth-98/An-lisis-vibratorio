#hace un entrenamiento de los coores del cabl pero no de la etiqueta
import cv2
import pytesseract
import numpy as np
import re
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#legibilidad de la imagen
def legibility(image):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > 100 #parametro simulafo, ajustar si se necesita
#extraer el texto
def extraer_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.treshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary, config='--psm 6')
    return text.sprit()
# validar elfotmato
def validate_serial(serial_number):
    pattern = r'^BCC0577BCCM425-0000-1A-06-PX0334-100' #CHECAR QUE PARAMETROS METER CORRECTAMENTE
    return bool(re.match(pattern, serial_number))

#vision en timpo real
def realtime_detection(color_model_path='clasificacionColor.h5', labels=None):
    color_model = load_model(color_model_path)
    if not labels:
        raise ValueError("debees proporcionar la etiqueta de los colores")
    
    #iniciar captura
    cap=cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video")
            break 
        
        if not legibility(frame):
         result ="imagen borrosa. Intente ajustar la amara"
        else: 
        #extrer y validar numero de serie
            serial_text = extraer_text(frame)
        if not serial_text:
            result= "No se pudo leer el numero de serie"
        elif not validate_serial(serial_text):
            result = f'Numero de serie invalido: {serial_text}'
        else:
            result = f'Numero de serie valido: {serial_text}'


        #mostrar el resultado
        cv2.putText(frame, result, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Validacion de numero de serie', frame)

        #salir con la tecla q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    validate_serial()
        