import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import json
import random
import pickle

#nltk.download('punkt')

#importare database
with open("training_set.json") as archivio:
    dati = json.load(archivio)

parlare = []
tags = []
auxX = []
auxY = []

#indicizzare il contenuto
for contenuto in dati["intents"]:
    for patterns in contenuto["patterns"]:
        auxParlare = nltk.word_tokenize(patterns)
        parlare.extend(auxParlare)
        auxX.append(auxParlare)
        auxY.append(contenuto["tag"])

        if contenuto["tag"] not in tags:
            tags.append(contenuto["tag"])

parlare = [stemmer.stem(w.lower()) for w in parlare if w != "?"]
parlare = sorted(list(set(parlare)))
tags = sorted(tags)

#creare una lista di indici nell'array
entrare = []
salita = []

salitaV = [0 for _ in range(len(tags))]

for x, documento in enumerate(auxX):
    recipiente = []
    auxParlare = [stemmer.stem(w.lower()) for w in documento]
    for w in parlare:
        if w in auxParlare:
            recipiente.append(1)
        else:
            recipiente.append(0)
    Fsalita = salitaV[:]
    Fsalita[tags.index(auxY[x])] = 1
    entrare.append(recipiente)
    salita.append(Fsalita)

#Rete Neuronale

entrare = numpy.array(entrare)
salita = numpy.array(salita)

#creazione neuroni

tensorflow.compat.v1.reset_default_graph()


red = tflearn.input_data(shape=[None, len(entrare[0])])
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, len(salita[0]), activation="softmax")
red = tflearn.regression(red)

#creazione modello
model = tflearn.DNN(red)
model.fit(entrare, salita, n_epoch=1000, batch_size=18, show_metric=True)
model.save("Model.tflearn")

# continua: https://www.youtube.com/watch?v=2Ty86O6dzts


def mainBot():
    print("Sono 'BOT', qui per farti compagnia...")
    while True:
        entrata = input("Tu: ")
        recipiente = [0 for _ in range(len(parlare))]
        entrataProcessata = nltk.word_tokenize(entrata)
        entrataProcessata = [stemmer.stem(parlata.lower()) for parlata in entrataProcessata]
        for parlataIndividuale in entrataProcessata:
            for i, parlata in enumerate(parlare):
                if parlata == parlataIndividuale:
                    recipiente[i] = 1
        risultato = model.predict([numpy.array(recipiente)])
        risultatoIndice = numpy.argmax(risultato)
        tag = tags[risultatoIndice]

        for tagAux in dati["intents"]:
            if tagAux["tag"] == tag:
                risposta = tagAux["responses"]

        print("BOT: ", random.choice(risposta))
mainBot()

