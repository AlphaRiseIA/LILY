# LILY
#AlphaRiseIA

import requests
import random
import numpy
import tflearn
import tensorflow
import json
import pickle
import nltk
import time
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("contenido_lily.json") as archivo:
    datos_lily = json.load(archivo)

try:
    with open("datos_lily.pickle", "rb") as f:
        palabras_lily, tags_lily, entrenamiento_lily, salida_lily = pickle.load(f)
except:
    palabras_lily = []
    tags_lily = []
    auxX = []
    auxY = []

    for contenido in datos_lily["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras_lily.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])
            if contenido["tag"] not in tags_lily:
                tags_lily.append(contenido["tag"])

    palabras_lily = [stemmer.stem(w.lower()) for w in palabras_lily if w != "?"]
    palabras_lily = sorted(list(set(palabras_lily)))
    tags_lily = sorted(tags_lily)

    entrenamiento_lily = []
    salida_lily = []

    salidaVacia = [0 for _ in range(len(tags_lily))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras_lily:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[tags_lily.index(auxY[x])] = 1
        entrenamiento_lily.append(cubeta)
        salida_lily.append(filaSalida)

    entrenamiento_lily = numpy.array(entrenamiento_lily)
    salida_lily = numpy.array(salida_lily)

    with open("datos_lily.pickle", "wb") as f:
        pickle.dump((palabras_lily, tags_lily, entrenamiento_lily, salida_lily), f)

tensorflow.compat.v1.reset_default_graph()

#--------------------------------------------------------------------------------------------------------
red = tflearn.input_data(shape=[None, len(entrenamiento_lily[0])])
red = tflearn.fully_connected(red, 50)
red = tflearn.fully_connected(red, 50)
red = tflearn.fully_connected(red, len(salida_lily[0]), activation="softmax")
red = tflearn.regression(red)
#--------------------------------------------------------------------------------------------------------

modelo_lily = tflearn.DNN(red)

try:
    modelo_lily.load("modelo_lily.tflearn")
except:
    modelo_lily.fit(entrenamiento_lily, salida_lily, n_epoch=5000, batch_size=8, show_metric=True)
    modelo_lily.save("modelo_lily.tflearn")
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
chatbots = {
    'Zeus': {'url': 'http://localhost:5001/chat', 'disponible': False},
    'Hera': {'url': 'http://localhost:5002/chat', 'disponible': False},
    'Poseidón': {'url': 'http://localhost:5003/chat', 'disponible': False},
    'Atenea': {'url': 'http://localhost:5004/chat', 'disponible': False},
    'Apolo': {'url': 'http://localhost:5005/chat', 'disponible': False},
    'Hades': {'url': 'http://localhost:5006/chat', 'disponible': False}
}

chatbot_actual = 'LILY'

def verificar_disponibilidad_chatbots():
    print("Verificando disponibilidad de los chatbots...")
    for nombre, info in chatbots.items():
        try:
            response = requests.get(info['url'].replace('/chat', '/status'), timeout=2)
            if response.status_code == 200:
                info['disponible'] = True
                print(f"{nombre} está disponible.")
            else:
                info['disponible'] = False
                print(f"{nombre} no está disponible.")
        except:
            info['disponible'] = False
            print(f"{nombre} no está disponible.")
    print("Verificación completada.\n")

def LILY(entrada):
    cubeta = [0 for _ in range(len(palabras_lily))]
    entradaProcesada = nltk.word_tokenize(entrada)
    entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
    for palabraIndividual in entradaProcesada:
        for i, palabra in enumerate(palabras_lily):
            if palabra == palabraIndividual:
                cubeta[i] = 1
    resultados = modelo_lily.predict([numpy.array(cubeta)])
    resultados_indices = numpy.argmax(resultados)
    tag = tags_lily[resultados_indices]

    for tagAux in datos_lily["contenido"]:
        if tagAux["tag"] == tag:
            respuestas = tagAux["respuestas"]
            if tagAux["tag"] == "cambiar_chatbot":
                return random.choice(respuestas), True
    return random.choice(respuestas), False

def obtener_respuesta_chatbot(nombre_chatbot, mensaje):
    url = chatbots[nombre_chatbot]['url']
    try:
        respuesta = requests.post(url, json={'mensaje': mensaje}, timeout=5).json()
        return respuesta['respuesta']
    except Exception as e:
        return f"Error al comunicarse con {nombre_chatbot}: {e}"

def main():
    global chatbot_actual
    verificar_disponibilidad_chatbots()
    print("¡Hola! Soy LILY. Puedes hablar conmigo o pedir cambiar a otro chatbot.")
    while True:
        mensaje = input(">>: ")
        if mensaje.lower() in ['salir', 'exit']:
            print("LILY: ¡Hasta luego!")
            break
        if chatbot_actual == 'LILY':
            respuesta, cambiar = LILY(mensaje)
            if cambiar:
                chatbot_encontrado = False
                for nombre in chatbots.keys():
                    if nombre.lower() in mensaje.lower():
                        chatbot_encontrado = True
                        if chatbots[nombre]['disponible']:
                            chatbot_actual = nombre
                            print(f"LILY: Cambiando a {chatbot_actual}.")
                        else:
                            print(f"LILY: Lo siento, {nombre} no está disponible en este momento.")
                        break
                if not chatbot_encontrado:
                    print("LILY: No especificaste un chatbot válido. Por favor, indica uno de los siguientes:")
                    chatbots_disponibles = [nombre for nombre in chatbots.keys()]
                    print(", ".join(chatbots_disponibles))
            else:
                print(f"LILY: {respuesta}")
        else:
            if mensaje.lower() in ['volver', 'regresar', 'cambiar a lily']:
                chatbot_actual = 'LILY'
                print("LILY: Estás de vuelta conmigo.")
            else:
                if chatbots[chatbot_actual]['disponible']:
                    respuesta = obtener_respuesta_chatbot(chatbot_actual, mensaje)
                    print(f"{chatbot_actual}: {respuesta}")
                else:
                    print(f"LILY: Lo siento, {chatbot_actual} no está disponible en este momento.")
                    chatbot_actual = 'LILY'
                    print("LILY: Estás de vuelta conmigo.")

if __name__ == '__main__':
    main()
