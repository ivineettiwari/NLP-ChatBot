#Imports
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

#Loading Data
with open("files/final.json") as file:
    data = json.load(file)

# Load data from disk
with open("files/data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("files/model.tflearn")


#Function to process input
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    stemmer = LancasterStemmer()
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

seat_count = 50

@app.route('/get')
def get_bot_response():
    global seat_count
    message = request.args.get('msg')
    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message,words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        response = random.choice(responses)
    else:
        response = "I didn't quite get that, please try again."
    return response


if __name__ == "__main__":
	app.run(port=8000, debug=True)