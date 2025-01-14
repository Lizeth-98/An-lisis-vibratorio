import paho.mqtt.client as mqtt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import time
 
# Configuración global
datos_mensuales = []
 
# Inicializar modelo
modelo = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
escalador = StandardScaler()
 
# Función al recibir un mensaje MQTT
def on_message(client, userdata, msg):
    global datos_mensuales
    try:
        # Decodificar mensaje
        mensaje = msg.payload.decode("utf-8")
        x, y, z, temperatura = map(float, mensaje.split(","))
        
        # Determinar si es falla
        falla = (abs(x) > 8) | (abs(y) > 8) | (abs(z) > 8) | (temperatura > 80)
        etiqueta = 1 if falla else 0
        
        # Guardar datos recibidos
        datos_mensuales.append([x, y, z, temperatura, etiqueta])
        print(f"Datos recibidos: X={x}, Y={y}, Z={z}, T={temperatura}, Falla={etiqueta}")
        
    except Exception as e:
        print(f"Error al procesar mensaje: {e}")
 
# Configurar el cliente MQTT
def setup_mqtt():
    client = mqtt.client()
    client.on_message = on_message
    client.username_pw_set("mqtt", "!cQ6NzzuHWboGkMy!2KJTEjfDVJkgAe@ZuT8")
    try:
        # Conexión al broker MQTT
        broker = "10.66.38.4"  # Host
        port = 1883
        client.connect(broker, port, 60)
        print("Se conectó de forma adecuada")
    except Exception as e:
        print(f"No se conectó: {e}")
 
# Función para mostrar gráfica
def mostrar_grafica(datos):
    df = pd.DataFrame(datos, columns=['X', 'Y', 'Z', 'Temperatura', 'Falla'])
    plt.figure(figsize=(12, 6))
    plt.plot(df['X'], label='Eje X', color='b')
    plt.plot(df['Y'], label='Eje Y', color='g')
    plt.plot(df['Z'], label='Eje Z', color='r')
    plt.plot(df['Temperatura'], label='Temperatura', color='orange')
    plt.title("Tendencias de Vibración y Temperatura")
    plt.xlabel("Tiempo")
    plt.ylabel("Valores")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(8, color='gray', linestyle='--', label='Límite de Alerta')
    plt.axhline(-8, color='gray', linestyle='--')
    plt.legend()
plt.show()
 
# Simulación de entrenamiento y visualización
for mes in range(1, 5):  # Simulación de 4 meses
    print(f"\n>> Iniciando captura de datos del mes {mes}...\n")
    client.loop_start()
 
    # Simulación de un mes (espera de 10 segundos para pruebas)
    time.sleep(10)
    client.loop_stop()
 
    if len(datos_mensuales) > 0:
        # Mostrar gráfica de tendencias
        mostrar_grafica(datos_mensuales)
        
        # Crear DataFrame y preprocesar datos
        datos = pd.DataFrame(datos_mensuales, columns=['X', 'Y', 'Z', 'Temperatura', 'Falla'])
        X_nuevo = datos[['X', 'Y', 'Z', 'Temperatura']]
        y_nuevo = datos['Falla']
 
        if mes == 1:
            X_nuevo_escalado = escalador.fit_transform(X_nuevo)
            modelo.partial_fit(X_nuevo_escalado, y_nuevo, classes=[0, 1])
        else:
            X_nuevo_escalado = escalador.transform(X_nuevo)
            modelo.partial_fit(X_nuevo_escalado, y_nuevo)
 
        # Evaluación del modelo
        predicciones = modelo.predict(X_nuevo_escalado)
        precision = accuracy_score(y_nuevo, predicciones)
        print(f"\n>> Precisión después del mes {mes}: {precision*100:.2f}%")
        datos_mensuales.clear()  # Limpiar datos después de entrenar
 
# Prueba de predicción
nuevos_datos_prueba = np.array([[5.0, 5.0, 6.0, 70.0]])
nuevos_datos_prueba_escalados = escalador.transform(nuevos_datos_prueba)
prediccion = modelo.predict(nuevos_datos_prueba_escalados)
estado = "Falla" if prediccion[0] == 1 else "Normal"
print(f"\nPredicción para nuevos datos: {estado}")