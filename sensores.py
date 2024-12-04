import json
import paho.mqtt.client as mqtt
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
 
# Variables para almacenar las lecturas
humidity_data = []
temperature_data = []
pressure_data = []
 
# Configurar la red neuronal
scaler = StandardScaler()
model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=500, random_state=42)
 
# Datos simulados iniciales para hacer el entrenamiento
initial_data = np.array([
    [21.0, 25.0, 1013.0],
    [30.0, 28.0, 1010.0],
    [20.0, 22.0, 1015.0],
    [27.0, 30.0, 1008.0],
])
initial_labels = np.array([0, 1, 0, 1])  # 0: normal, 1: anomalía
 
# Ajustar el modelo con datos iniciales
scaled_data = scaler.fit_transform(initial_data)
model.fit(scaled_data, initial_labels)
 
# Función llamada cuando el cliente recibe un mensaje del servidor MQTT
def on_message(client, userdata, msg):
    global humidity_data, temperature_data, pressure_data
 
    # Decodificar mensaje
    payload = msg.payload.decode("utf-8")
    try:
        # Convertir el payload de JSON a diccionario
        payload_json = json.loads(payload)
        
        # Extraer y convertir el valor, verificando si está correctamente formateado
        if "value" in payload_json:
            value = float(payload_json["value"])
        else:
            raise KeyError("Campo 'value' no encontrado en el payload")
 
        # Procesar según el tópico
        topic = msg.topic
        if topic == "Balluff/AGS/Production/BNI/Edge/Environment/ambient_humidity":
            humidity_data.append(value)
        elif topic == "Balluff/AGS/Production/BNI/Edge/Environment/ambient_temperature":
            temperature_data.append(value)
        elif topic == "Balluff/AGS/Production/BNI/Edge/AirPressure1/pressure":
            pressure_data.append(value)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error al procesar el mensaje: {e}\nPayload: {payload}")
 
# Configurar el cliente MQTT
def setup_mqtt():
    client = mqtt.Client()
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
    
    # Suscripción a los tópicos
    client.subscribe("Balluff/AGS/Production/BNI/Edge/Environment/ambient_humidity")
    client.subscribe("Balluff/AGS/Production/BNI/Edge/Environment/ambient_temperature")
    client.subscribe("Balluff/AGS/Production/BNI/Edge/AirPressure1/pressure")
    return client
 
# Función para procesar datos y graficar
def process_data():
    global humidity_data, temperature_data, pressure_data
    while True:
        # Esperar 15 minutos (900 segundos)
        time.sleep(200)
 
        # Calcular promedios si hay datos
        avg_humidity = np.mean(humidity_data) if humidity_data else None
        avg_temperature = np.mean(temperature_data) if temperature_data else None
        avg_pressure = np.mean(pressure_data) if pressure_data else None
 
        # Limpiar las listas
        humidity_data.clear()
        temperature_data.clear()
        pressure_data.clear()
 
        if avg_humidity is not None and avg_temperature is not None and avg_pressure is not None:
            # Normalizar los datos para la red neuronal
            input_data = scaler.transform([[avg_humidity, avg_temperature, avg_pressure]])
            # Predicción del modelo
            prediction = model.predict(input_data)[0]
            status = "Normal" if prediction < 0.5 else "Anomalía"
 
            # Mostrar resultados
            print(f"Promedio de los últimos 15 minutos:")
            print(f"- Humedad: {avg_humidity:.2f}")
            print(f"- Temperatura: {avg_temperature:.2f}")
            print(f"- Presión: {avg_pressure:.2f}")
            print(f"Analisis predictivo de la red neuronal: {status}")
            print("----------------------------")
 
            # Graficar datos
            plt.figure(figsize=(10, 6))
            plt.bar(["Humedad", "Temperatura", "Presión"], [avg_humidity, avg_temperature, avg_pressure], color=["blue", "red", "green"])
            plt.title("Promedios de sensores en los últimos 15 minutos")
            plt.ylabel("Valores Promedio")
            plt.xlabel("Sensores")
            plt.ylim(0, max(avg_humidity, avg_temperature, avg_pressure) + 10)
            plt.show()
        else:
            print("No se recibieron suficientes datos en los últimos 15 minutos.")
 
if __name__ == "__main__":
    client = setup_mqtt()
    client.loop_start()  # Inicializar el loop para recibir datos
    try:
        process_data()
    except KeyboardInterrupt:
        print("Deteniendo el programa...")
        client.loop_stop()
        client.disconnect()