# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import json
import random
with open('final_intents_data.json') as json_data:

    intents = json.load(json_data)

stemmer =  LancasterStemmer()

model = load_model('my_model.h5')       

global graph
graph = tf.compat.v1.get_default_graph()

filename = 'final_intents_data.pkl'
model_2 = pickle.load(open(filename, 'rb'))

words = model_2['words']
classes = model_2['classes']

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/', methods=['POST'])
       
def reply():
    sentence_in = request.form['msg'] 
    sentence_words_1 = nltk.word_tokenize(sentence_in)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words_1]
    # bag of words
    bag = [0]*len(words)  

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    
        # generate probabilities from the model
    input_data = pd.DataFrame([np.array(bag)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
        
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]]})
        # return tuple of intent and probability
    intnt = return_list[0]['intent']

    #resp = [d['responses'][np.random.randint(0,len(d['responses']))]  for d in intents['intents'] if d['tag'] == intnt]
    for d in intents['intents']:

      if d['tag'] == intnt:

        l = len(d['responses'])

        ind = np.random.randint(0,l)

        response = d['responses'][ind]

    #responses = return_list
    return render_template('index.html', user_in = sentence_in, resp=response)

if __name__ == '__main__':
	app.run(debug=True)
