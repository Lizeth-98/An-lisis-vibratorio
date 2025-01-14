import cv2
import numpy as np

#cargar la imgen para veriifcar la forma de la cabea del cable
pl_cabeza= cv2.imread("direccion de la imagen.png", 0)

#configuracion de la camara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("error al capturar la imagen")
        break
    
    #convertir a la escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #verificar si el cable esta coreectamente cargado en la placa
    edges = cv2.Canny(gray, 50, 150) #deteccion de los bordes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cable_corect = False
    for cnt in contours:
        #analizar contornos como area o posiscion
        area = cv2.contourArea(cnt)
        if 500 < area < 3000: #datos simulados ajsutar el tamaño dpendiendo del area de la placa
            x, y, w, h = cv2.boundingRect(cnt)
            if x > 100 and y > 100: #ejemplo de posicion, ajustar segun el cable 
                cv2.rectangle(frame, (x,y), (x + w, y +h), (255, 0, 0), 2)
                cable_corect = True
                break
        
    if cable_corect:
        cv2.putText(frame, "Cable colocado correctamente", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
     cv2.putText(frame, "Cable colocado incorrectamente", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #DETECCION DE LA CABEZA DEL CABLE
    res = cv2.matchTemplate(gray, pl_cabeza, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #si el valor maximo supera el umbral la cabeza esta moldeada correctamente
    threshold = 0.8
    if  max_val >= threshold:
        cv2.rectangle(frame, max_loc,
                  (max_loc[0] + pl_cabeza.shape[1], max_loc[1] + pl_cabeza.shape[0]),
                  (0, 255, 0), 2)
        cv2.putText(frame, "Cabeza correcta", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Cabeza incorrecta", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    #mostrar los resultados
    cv2.show("Moldeo", frame)

    #salir con la tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
