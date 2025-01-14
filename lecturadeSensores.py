import paho.mqtt.client as mqtt
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
 
# variables para almacenar las lecturas
humidity_data = []
temperature_data = []
pressure_data = []
 
#configurar la red neuroal
scaler = StandardScaler()
model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=500, random_state=42)
 
# datos simulados inisiales para hacer el entrenamiento
initial_data = np.array([
    [50.0, 25.0, 1013.0],
    [60.0, 28.0, 1010.0],
    [45.0, 22.0, 1015.0],
    [70.0, 30.0, 1008.0],
])
initial_labels = np.array([0, 1, 0, 1])  # 0: normal, 1: Anomalia
 
# ajustar el modelo con datos iniciales
scaled_data = scaler.fit_transform(initial_data)
model.fit(scaled_data, initial_labels)
 
# esta fununcion es llamada cuando el cliente recibe un mensaje del servidor mqtt
def on_message(client, userdata, msg):
    global humidity_data, temperature_data, pressure_data
 
    # decodificar mensaje
    payload = msg.payload.decode("utf-8")
    topic = msg.topic
 
    if topic == "Balluff/AGS/Production/BNI/Edge/Environment/ambient_humidity":
        humidity_data.append(float(payload))
    elif topic == "Balluff/AGS/Production/BNI/Edge/Environment/ambient_temperature":
        temperature_data.append(float(payload))
    elif topic == "Balluff/AGS/Production/BNI/Edge/AirPressure1/pressure":
        pressure_data.append(float(payload))
 
# configurar el cliente mqtt
def setup_mqtt():
    client = mqtt.Client()
    client.on_message = on_message
 
    # Conexion al broker qtt
    broker = "10.66.38.4" #host
    port = 1883
    client.connect(broker, port, 60)
 
    # suscripcion al tipic
    client.subscribe("Balluff/AGS/Production/BNI/Edge/Environment/ambient_humidity")
    client.subscribe("Balluff/AGS/Production/BNI/Edge/Environment/ambient_temperature")
    client.subscribe("Balluff/AGS/Production/BNI/Edge/AirPressure1/pressure")
 
    return client
 
# Función para procesar datos y graficar
def process_data():
    global humidity_data, temperature_data, pressure_data
 
    while True:
        # Esperar 15 minutos (900 segundos)
        time.sleep(900)
 
        # Calcular promedios si hay datos
        avg_humidity = np.mean(humidity_data) if humidity_data else None
        avg_temperature = np.mean(temperature_data) if temperature_data else None
        avg_pressure = np.mean(pressure_data) if pressure_data else None
 
        # Limpiar las listas
        humidity_data.clear()
        temperature_data.clear()
        pressure_data.clear()
 
        if avg_humidity is not None and avg_temperature is not None and avg_pressure is not None:
            # normalizar los datos para la red neuronal
            input_data = scaler.transform([[avg_humidity, avg_temperature, avg_pressure]])
            
            # Predicción del modelo
            prediction = model.predict(input_data)[0]
            status = "Normal" if prediction < 0.5 else "Anomalía"
            
            # Mostrar resultados
            print(f"Promedio de los últimos 15 minutos:")
            print(f"- Vibración: {avg_humidity:.2f}")
            print(f"- Temperatura: {avg_temperature:.2f}")
            print(f"- Presión: {avg_pressure:.2f}")
            print(f"Estado predicho por la red neuronal: {status}")
            print("----------------------------")
 
            # graficar datos
            plt.figure(figsize=(10, 6))
            plt.bar(["Vibración", "Temperatura", "Presión"], [avg_humidity, avg_temperature, avg_pressure], color=["blue", "red", "green"])
            plt.title("Promedios de sensores en los últimos 15 minutos")
            plt.ylabel("Valores Promedio")
            plt.xlabel("Sensores")
            plt.ylim(0, max(avg_humidity, avg_temperature, avg_pressure) + 10)
            plt.show()
 
    else:
        print("No se recibieron suficientes datos en los últimos 15 minutos.")
 
if __name__ == "__main__":
    client = setup_mqtt()
    client.loop_start()  # inicializar elloop para recibir datos
 
    try:
        process_data()
    except KeyboardInterrupt:
        print("Deteniendo el programa...")
        client.loop_stop()
        client.disconnect()
