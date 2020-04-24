#chatbot  AI using Nural Networks and deep learning
#uses intents json should be in same folder

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import pickle

import numpy
import tflearn
import json
import tensorflow #deeplearning
import random


with open("intents.json") as file:
    data = json.load(file) #use intents from that file


#print (data)
#print (data["intents"])

try:
    #storing and opening the data
    #try so we do not have to train bot everytime
    #rb read binary
    with open("data.pickle","rb") as f:
        words, labels ,training , output = pickle.load(f) #storing all these in a file
except:
    #

    words = []
    labels = []
    docs_x = [] 
    docs_y = []

    #loop through intents

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds) #what intent it is
            docs_y.append(intent["tag"]) #what tag t is a part of


        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    #stem all words and remove duplicates

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)


    #as neural newtork understands only numbrs we create a list which store frequency of the words

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) #word exists
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    #convert to array
    training = numpy.array(training)
    output = numpy.array(output)

    #saving
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels ,training , output), f) 





tensorflow.reset_default_graph() #resetibg graph


# X
net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8) #8 neurons
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

#to train
model = tflearn.DNN(net)

#n_epoch  means the no of time the model will see the same data the more the time the better it can understand


try:
    model.load("model.tflearn")
except:    
    model.fit(training,output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s , words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    #generate bag word 0 1 
    for se in s_words:
        for i , w in enumerate(words):
            if w == se:
                bag[i] = 1  #if exists
    
    return numpy.array(bag)


def chat():
    print("Start talking with the bot! (type quit to stop) ")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        #sending input to the chatbot
        results = model.predict([bag_of_words(inp,words)])
        #print(results)  prints the probability
        results_index = numpy.argmax(results) #returns the index with the max prob
        tag = labels[results_index] #the tag

        #to find the tag and return the response

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        print(random.choice(responses))


chat()