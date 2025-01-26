#Imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle 
import json

nltk.download('punkt')


def my_jobs():
    # your code here
    #Loading Data
    with open("files/final.json") as file:
        data = json.load(file)
    
    #Initializing empty lists
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #Looping through our data
    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern = pattern.lower()
                #Creating a list of words
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    stemmer = LancasterStemmer()
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    for x,doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    #Converting training data into NumPy arrays
    training = np.array(training)
    output = np.array(output)

    #Saving data to disk
    with open("files/data.pickle","wb") as f:
        pickle.dump((words, labels, training, output),f)
    

    tf.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape = [None, len(training[0])])
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    ## Train the model
    model.fit(training, output, n_epoch = 400, batch_size = 20, show_metric = True)
    model.save("files/model.tflearn")
    pass


if __name__ == '__main__':
    my_jobs()