import json
import pickle
import numpy as np 
import re
import unicodedata
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay

# Función para normalizar y tokenizar
def normalizar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return re.findall(r'\b\w+\b', texto)

# Cargar el archivo JSON de intenciones
with open('Informacion.json', 'r', encoding='utf-8') as archivo:
    Intentos = json.load(archivo)

Palabras = []
Clases = []
Documentos = []
IgnorarPalabras = ['.', ',', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '¿', '?', '¡', '!']

# Recorre cada intento y sus patrones
for intento in Intentos['intents']:
    for patron in intento['preguntas']:       
        palabras_tokenizadas = normalizar_texto(patron)
        Palabras.extend(palabras_tokenizadas)
        Documentos.append((palabras_tokenizadas, intento['tag']))
        if intento["tag"] not in Clases:
            Clases.append(intento['tag'])

# Eliminar duplicados y ordenar
Palabras = [p for p in Palabras if p not in IgnorarPalabras]
Palabras = sorted(list(set(Palabras)))
Clases = sorted(list(set(Clases)))

# Guardar vocabulario y clases
pickle.dump(Palabras, open('words.pkl', 'wb'))
pickle.dump(Clases, open('classes.pkl', 'wb'))

Entrenamiento = []
SalidaVacia = [0] * len(Clases)

for doc in Documentos:
    patron_palabras = doc[0]
    bolsa = [1 if palabra in patron_palabras else 0 for palabra in Palabras]
    salida = list(SalidaVacia)
    salida[Clases.index(doc[1])] = 1
    Entrenamiento.append([bolsa, salida])

random.shuffle(Entrenamiento)

EntrenamientoX = np.array([fila[0] for fila in Entrenamiento])
EntrenamientoY = np.array([fila[1] for fila in Entrenamiento])

# Crear el modelo
modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(EntrenamientoX[0]),), activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(EntrenamientoY[0]), activation='softmax'))

# Configurar el optimizador
tasa_aprendizaje = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

optimizador = SGD(learning_rate=tasa_aprendizaje, momentum=0.9, nesterov=True)
modelo.compile(loss='categorical_crossentropy', optimizer=optimizador, metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(EntrenamientoX, EntrenamientoY, epochs=200, batch_size=5, verbose=1)

# Guardar modelo
modelo.save('chatbot_model.keras')

print("✅ Modelo entrenado y guardado con éxito")
