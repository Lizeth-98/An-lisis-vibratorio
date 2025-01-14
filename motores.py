import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
# Simulación de datos de vibración y temperatura
def generar_datos(n=1000):
    x = np.random.uniform(-10, 10, n)
    y = np.random.uniform(-10, 10, n)
    z = np.random.uniform(-10, 10, n)
    temperatura = np.random.uniform(20, 100, n)
    
    # Etiquetas: 0 = Normal, 1 = Falla
    falla = (abs(x) > 8) | (abs(y) > 8) | (abs(z) > 8) | (temperatura > 80)
    etiquetas = np.where(falla, 1, 0)
    
    datos = pd.DataFrame({
        'X': x, 'Y': y, 'Z': z, 'Temperatura': temperatura, 'Falla': etiquetas
    })
    return datos
 
# Generar datos simulados
datos = generar_datos()
 
# Visualización de datos
plt.scatter(datos['Temperatura'], datos['Z'], c=datos['Falla'], cmap='coolwarm')
plt.title("Distrubuion de los datos")
plt.xlabel("Temperatura")
plt.ylabel("Vibración Z")
plt.show()
 
# Separar datos
X = datos[['X', 'Y', 'Z', 'Temperatura']]
y = datos['Falla']
 
# Preprocesamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
escalador = StandardScaler()
X_train = escalador.fit_transform(X_train)
X_test = escalador.transform(X_test)
 
# Crear modelo de red neuronal
modelo = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
 
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Entrenar modelo
modelo.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
 
# Evaluar modelo
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f"Precisión del modelo: {accuracy*100:.2f}%")
 
# Prueba de predicción
nuevos_datos = np.array([[5.0, 5.0, 6.0, 70.0]])
nuevos_datos_escalados = escalador.transform(nuevos_datos)
prediccion = modelo.predict(nuevos_datos_escalados)
estado = "Falla" if prediccion >= 0.5 else "Normal"
print(f"Predicción para nuevos datos: {estado}")