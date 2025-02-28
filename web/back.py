from flask import Flask, request, render_template, jsonify
import openai
from openai import OpenAI
import numpy as np  # Assuming you are using numpy for embedding processing
import os
import random
import json, time, pickle
import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


model_names = {
    "1": "background_nn.keras",
    "2": "significance_svc.model",
    "3": "aims_gpt.keras",
    "4": "methods_bilstm_pca.keras",
    "5": "outcome_nn.keras",
    "6": "resource_svc.model",
    "7": "conclusion_svc.model",
    "8": "members_rnn.keras"
}

seg_names = [
  "background",
  "significance",
  "aims",
  "methods",
  "outcome",
  "resource",
  "conclusion",
  "members"]

# Function to get embeddings from GPT
def get_embeddings(text):
    file_path = 'novo_openai.jsonl'
    # Open a file in write mode
    with open(file_path, 'w') as file:
        for t in text:
            # Convert the dictionary to a JSON string
            # and write it to the file with a newline character
            if len(text[t])==0:continue
            json_string = json.dumps({
                "custom_id": str(t),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": "text-embedding-ada-002",
                        "input": text[t]
                        }
            })
            file.write(json_string + '\n')
            
            
    batch_input_file = client.files.create(
      file=open("novo_openai.jsonl", "rb"),
      purpose="batch"
    )
    
    batch_input_file_id = batch_input_file.id
    

    dd = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
          "description": "nightly eval job"
        }
    )
    print(dd.id)
    
    time.sleep(10)
    d = client.batches.retrieve(dd.id)
    while d.status != "completed":
      time.sleep(10)
      d = client.batches.retrieve(dd.id)
    
    print(d)
    file_response = client.files.content(d.output_file_id)

    a = file_response.text
    b = a.split("\n")
    res_emb = {}
    for bb in b:
        try:
            cur_emb = json.loads(bb)
            res_emb[cur_emb["custom_id"]] = cur_emb["response"]["body"]["data"][0]["embedding"]
        except:
            continue
    
    return res_emb

# Function to process embeddings and return values for frontend
def process_embeddings(embeddings):
    # Do some processing on the embeddings here
    scores = {}
    for k in embeddings:
        model_name = model_names[k]
        if "keras" in model_name:
            model = keras.models.load_model("models/"+model_name)
        else:
            m = open("models/"+model_name, "rb")
            model = pickle.load(m)
            m.close()
        
        X = np.array(embeddings[k])
        scaler = pickle.load(open("models/"+seg_names[int(k)-1]+"_scaler.model",
                                "rb"))
        X = scaler.transform(X.reshape(1, -1))
        print(X.shape)
        if "pca" in model_name:
            pca = pickle.load(open("models/methods_pca.model", "rb"))
            X = pca.transform(X)

        # reshape
        X = np.array(X).reshape((1, 1, -1)) 
        print(k, X.shape)
        if "svc" in model_name:
          scores[seg_names[int(k)-1]] = float(model.predict_proba(X.reshape(len(X), -1))[:,1][0])
        else:
          scores[seg_names[int(k)-1]] = float(model.predict(X)[0][0])

    return scores 

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing the input
@app.route('/process', methods=['POST'])
def process_segments():
    segments = {}
    for i in range(1, 9):
        segments[i] = request.form.get(f'segment{i}')

    # Get embeddings for each non-empty segment
    embeddings = get_embeddings(segments)

    # Process the embeddings and return 10 values
    processed_values = process_embeddings(embeddings)
    print(processed_values)
    print(jsonify(processed_values))

    # Return the processed values as a JSON response
    return jsonify(processed_values)

if __name__ == '__main__':
    app.run(debug=True)

