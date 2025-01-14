import cv2
import numpy as np
 
# procesar la imagen de referencia
def procesar_img(ruta_imagen): #poner aqui la inmagen que se procesara
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    original = imagen.copy()
 
    # convertir la imagen a la escla de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
 
    # eeducir el ruido de la escala de grises
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
 
    # detectar los bordes
    bordes = cv2.Canny(gris, 50, 150)
 
    # detectar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # DIBUJAR LOS CONTORNOS RN LA IMAGRN ORIGINAL
    cv2.drawContours(imagen, contornos, -1, (0, 255, 0), 2)
 
    # Crear un diccionario para guardar las propiedades de los sensores
    sen_ref = []
 
    for contorno in contornos:
        # Calcular el área del contorno
        area = cv2.contourArea(contorno)
 
        # Filtrar contornos pequeños o grandes
        if area < 100 or area > 10000:
            continue
 
        # Calcular el bounding box del contorno
        x, y, w, h = cv2.boundingRect(contorno)
 
        # Guardar las dimensiones como referencia
        sen_ref.append((x, y, w, h))
 
    return sen_ref
 
# Comparar sensores en tiempo real con la referencia
def monitorear_sensores(ruta_referencia):#rutade la imagen e referencia
    # Procesar la imagen de referencia y obtener sus sensores
    sen_ref = procesar_img(ruta_referencia)
 
    # Iniciar la cámara
    camara = cv2.VideoCapture(0)
 
    if not camara.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return
 
    while True:
        # Capturar un frame de la cámara
        ret, frame = camara.read()
        if not ret:
            print("Error: No se pudo capturar el frame.")
            break
 
        # Convertir el frame a escala de grises y detectar bordes
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(gris, 50, 150)
 
        # Detectar contornos en el frame actual
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        sen_correct = True
 
        for contorno in contornos:
            # Calcular el área del contorno
            area = cv2.contourArea(contorno)
            if area < 100 or area > 10000:
                continue
 
            # Calcular el bounding box del contorno
            x, y, w, h = cv2.boundingRect(contorno)
 
            # Comparar con los sensores de referencia
            match = False
            for ref_x, ref_y, ref_w, ref_h in sen_ref:
                if abs(ref_w - w) < 20 and abs(ref_h - h) < 20:
                    match = True
                    break
 
            # Dibujar rectángulos dependiendo si coinciden o no
            if match:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # verde: correcto
            else:
                sen_correct = False
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  #rojo: incorrecto
 
        # Mostrar mensaje en la imagen
        texto = "Sensores Correctos" if sen_correct else "Defectos Detectados"
        color = (0, 255, 0) if sen_correct else (0, 0, 255)
        cv2.putText(frame, texto, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
 
        # Mostrar el resultado
        cv2.imshow("Resultado en Tiempo Real", frame)
 
        # Salir al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # Liberar la cámara y cerrar las ventanas
    camara.release()
    cv2.destroyAllWindows()
 
# Ruta de la imagen de referencia
ruta_img = "sensordereferencia.jpg"  #aqi va la ruta de imagen de referencia
 
# Llamar a la función para monitorear
monitorear_sensores(ruta_img)