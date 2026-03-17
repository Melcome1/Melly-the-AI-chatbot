import random
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import numpy as np
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot.h5")

def clean_up_setence(setence):
    setence_words = nltk.word_tokenize(setence)
    setence_words = [lemmatizer.lemmatize(word) for word in setence_words]
    return setence_words

def bag_of_words(setence):
    word_setence = clean_up_setence(setence)
    bag = [0] * len(words)
    for w in word_setence:
        for i, word in enumerate(words):
            if word ==w:
                bag[i]=1

    return np.array(bag)

def predict_class(setence):
    bow = bag_of_words(setence)
    res = model.predict(np.array([bow]), verbose = 0)[0]

    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key = lambda x:x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})

    return return_list

def get_response(intent_list, intent_json):
    if not intent_list:
        default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "I don't have a response for that yet. Could you ask something else?",
            "That's interesting! Could you tell me more in a different way?"
        ]

        return random.choice(default_responses)
    
    tag = intent_list[o]['intent']
    list_of_intents = intent_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return "I'm still learning, can you ask that later"

print("\nHi I'm Melly, would be glad to start aconversation with you :)")

while True:
    message = input("\n:")
    if message.lower() in ['quit', 'exit', 'bye', 'goodbye']:
        print("Goodbye! It was nice chatting with you!")
        break
    
    ints = predict_class(message)
    res = (ints, intents)
    print(res)
