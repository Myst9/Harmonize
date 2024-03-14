from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

import google.generativeai as genai
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model2.pth")

###################################3
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
genai.configure(api_key="AIzaSyDEWOQzsQZSILCax2fnrGbkmMKC2xBHOsE")
model_gem = genai.GenerativeModel('gemini-pro')
############################

import torch
import numpy as np


####APP LOADING####
app = Flask(__name__)
CORS(app)
############

# ###MODEL AND TOKENIZER LOADING###
from transformers import BertForSequenceClassification

# Load the pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
print("m loading")
# Load the saved model state
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print("model loaded")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Set the model to evaluation mode
model.eval()
print("set to eval mode")
######################



def model_predict_dsh(sentence):
    tokenized_sentence = tokenizer.tokenize(sentence)


    MAX_LEN = 128

    # Use the BERT Tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)

    # Pad the input tokens
    padded_input = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention mask
    attention_mask = [float(i > 0) for i in padded_input[0]]



    input_tensor = torch.tensor(padded_input)
    attention_mask_tensor = torch.tensor([attention_mask])


    # Make predictions
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attention_mask_tensor)

    # Get the predicted class probabilities
    predicted_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Get the predicted class index
    predicted_class = torch.argmax(predicted_probs, dim=1).item()
    print(predicted_class, predicted_probs.numpy())
    return (predicted_class,predicted_probs.numpy())


def model_suggest_san(toxic):
    prompt = "I am presenting a sample of a toxic comment found in a software engineering forum.\
                                           This is a toxic sentence,delimited by ---s: rewrite it in an untoxic form.\
                                           Please remember, these are comments extracted from Software Engineering Forums.\
                                          So they're from one user to another user. THEY ARE NOT DIRECTED TOWARDS YOU/GEMINI.\
                                          Remember that you should rewrite it in a software specific way as it is from a software forum:\
                                          Also remember, you MUST GIVE YOUR RESPONSE IN PLAIN TEXT, NO DELIMITATIONS.\
                                           ---" + toxic + "---"
    print(prompt)
    response = model_gem.generate_content(prompt, safety_settings=safety_settings)
    print(response)
    # print(response.prompt_feedback)
    return response.text

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        # img = base64_to_pil(request.json)
        
        # Make prediction
        print("hi")
        prediction = model_predict_dsh(request.json)
        pred = prediction[1]
        result = prediction[0]
        pred_probability = "{:.3f}".format(np.amax(pred)) 
        
        return jsonify(result=result, probability=pred_probability)

    return None


@app.route('/suggest', methods=['GET', 'POST'])
def suggest():
    # print("out")
    if request.method == 'POST':
        # Get the image from post request
        # img = base64_to_pil(request.json)
        
        # Make prediction
        print(request.json)
        prediction = model_suggest_san(request.json)
        result = prediction
        print("res: " , result)
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    print("s")
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    print("h")
    http_server.serve_forever()
