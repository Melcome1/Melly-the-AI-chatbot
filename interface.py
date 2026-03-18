from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from ai_edge_litert.interpreter import Interpreter as tflite

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# Load chatbot data
intents = json.loads(open("intents.json", encoding="utf-8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Load TFLite model
interpreter = tflite(model_path="chatbot_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Prepare sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Convert sentence to bag of words
def bag_of_words(sentence):

    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)



# Predict intent
def predict_class(sentence):
    try:
        print(f"Predicting for: '{sentence}'")
        
        bow = bag_of_words(sentence)
        print(f"Bag of words shape: {bow.shape}")
        
        bow = np.array([bow]).astype(np.float32)
        print(f"Input shape: {bow.shape}")
        
        # Check input details
        print(f"Input details: {input_details}")
        print(f"Expected input shape: {input_details[0]['shape']}")
        
        interpreter.set_tensor(input_details[0]['index'], bow)
        interpreter.invoke()
        
        res = interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"Raw output: {res}")
        
        ERROR_THRESHOLD = 0.25
        
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        print(f"Results above threshold: {results}")
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        
        for r in results:
            return_list.append({
                "intent": classes[r[0]],
                "probability": float(r[1])
            })
        
        print(f"Return list: {return_list}")
        return return_list
        
    except Exception as e:
        print(f"❌ Error in predict_class: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


# Get chatbot response
def get_response(intent_list):

    if not intent_list:
        default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "I don't have a response for that yet. Could you ask something else?",
            "That's interesting! Could you tell me more in a different way?"
        ]
        return random.choice(default_responses)

    tag = intent_list[0]["intent"]

    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

    return "I'm still learning. Could you try asking something else?"


# Web interface
@app.route("/")
def home():
    return render_template("index.html")


# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():

    try:
        data = request.json
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"response": "Please type a message!"})

        if message.lower() in ["quit", "exit", "bye", "goodbye"]:
            return jsonify({"response": "Goodbye! It was nice chatting with you!"})

        ints = predict_class(message)

        response = get_response(ints)

        return jsonify({"response": response})

    except Exception as e:
        print("Error:", e)
        return jsonify({"response": "Sorry, I encountered an error processing your message."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
