import cv2
import numpy as np

#procesar la imagen 
def procesar_img():#aqui va la ruta de la imagen dntro del parentesis
    #leer la imagen
    imagen= cv2.imread()#ruta de la imagen que vamos a checar
    original = imagen.copy()
    
    #convertir la imagen a la escla de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    #reducir el ruido del color de la escala de grises
    gris = cv2.GaussianBlur(gris, (5,5), 0)
    #detectar los bordes
    bordes = cv2.Canny(gris, 50, 150)
    #detectrar contornos
    contornos = cv2.Canny.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #DIBUJAR LOS CONTORNOS RN LA IMAGRN ORIGINAL
    cv2.drawContours(imagen, contornos, -1, (0,255,0), 2)
    #evaluar la alineacion y el esamblaje del motor
    sen_correct = True
    for contorno in contornos:
        #calcular el area del contorno
        area = cv2.contourArea(contorno)
        #filtrar contornos grandes o pequeños, datos simulados por el momento
        if area < 100 or area > 10000:
            continue
        
        #calcular el boudingbox del contorno
        x,y,w,h = cv2.boundingRect(contorno)
        relacion_aspect = w / float(h)
        
        #evaluar las proporciones, datos simulados por el momento
        if relacion_aspect < 0.8 or relacion_aspect > 1.2:
            sen_correct = False
            cv2.rectangle(imagen, (x,y), (x+w, y+h), (0,0,255), 2) #el rojo es para el defecto
        else:
            cv2.rectangle(imagen, (x,y), (x+w, y+h), (0,255,0), 2) # y el verde para el correcto
            
    #mostrar los resultados
    texto = "Sensores Correctos" if sen_correct else "Defectos Detectados"
    color = (0, 255, 0) if sen_correct else (0, 0, 255)
    cv2.putText(imagen, texto, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    #mostrar imagen
    cv2.imshow("Imagen Original", original)
    cv2.imshow("Bordes Detectados", bordes)
    cv2.imshow("Resultado", imagen)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
#ruta de la img
ruta_img = "ruta.jpg" #aqui va la ruta de la imagen que sera procesada
procesar_img(ruta_img)
