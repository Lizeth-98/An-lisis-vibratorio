import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.processing import StandardScaler
from skalearn.neuronal_network import MLPClassifier
 

 #Datos simulados
def dato_vibracion(samples=1000):
    np.random.seed(45)
    frecuencies = np.random.uniform(50,100,samples)
    amplitudes = np.random.uniform(1,10,samples)
    noise = np.random.normal(0,0.5, samples)
    nivel_vibracion = amplitudes * np.sin(2*np.pi*frecuencies)+ noise

    #etiquetas 0 = normal, 1 = anomalo,
    labels = np.where(amplitudes>8,1,0)

    data = np.column_stack((frecuencies, amplitudes, nivel_vibracion, labels))
    return data

#gnrar los dts simulados
data = dato_vibracion()
X = data[:, :3]  #  Frecuencia, amplitud niveles de vibración
y = data[:, 3]   #  Etiqueta Normal o anomalo

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
#division datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=45)
 
# ntrenar el clasificador de la red neuronal
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=45)
model.fit(X_train, y_train)
 
# evaluacion modelo
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
 
#predecir en nuevos datos
new_data = np.array([[75, 6, 0.2]])  # Datos de ejemplo de frecuencia aplitud y nivel de vibracion
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"Prediction: {'Anomalo' if prediction == 1 else 'Normal'}")
 
# Visualizar resultados
anomalo = data[data[:, 3] == 1]
normal = data[data[:, 3] == 0]
 
plt.scatter(normal[:, 1], normal[:, 2], label='Normal', alpha=0.6)
plt.scatter(anomalo[:, 1], anomalo[:, 2], label='Anomalo', alpha=0.6, color='r')
plt.title('Vibration Analysis')
plt.xlabel('Amplitude')
plt.ylabel('Vibration Levels')
plt.legend()
plt.show()

