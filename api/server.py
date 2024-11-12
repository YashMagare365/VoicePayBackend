from flask import Flask, jsonify, request
import razorpay
from dotenv import load_dotenv
import os
from pyannote.audio import Model,Inference
from scipy.spatial.distance import cdist

# Load the model and create embeddings
model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")

load_dotenv()
app = Flask(__name__)

# Retrieve Razorpay credentials from environment variables
rzp_key = os.getenv('RAZORPAY_KEY')
secret_key = os.getenv('RAZORPAY_SECRET_KEY')

@app.route('/')
def index():
    print('hello world')
    return "Hello, World!"

@app.route('/members')
def members():
    return jsonify({"name": "John"})

# Uncomment the following route handlers if using nemo_toolkit
@app.route('/audio/embbed')
def audio():
    embedding1 = inference("/content/output2.wav")
    embedding2 = inference("/content/output3.wav")

    # Ensure embeddings are 2D (reshaped if necessary)
    embedding1 = embedding1.reshape(1, -1)  # Reshape to (1, D)
    embedding2 = embedding2.reshape(1, -1)  # Reshape to (1, D)

    # Calculate cosine distance between the two embeddings
    distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]

    print(f"Cosine distance between the embeddings: {distance}")

     
@app.route('/audiocheck/<string:s>',methods=['POST'])
def audiocheck(s):
    embedding1 = inference(s)
    embedding2 = inference(s)

    # Ensure embeddings are 2D (reshaped if necessary)
    embedding1 = embedding1.reshape(1, -1)  # Reshape to (1, D)
    embedding2 = embedding2.reshape(1, -1)  # Reshape to (1, D)

    # Calculate cosine distance between the two embeddings
    distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]

    print(f"Cosine distance between the embeddings: {distance}")
    res = str(distance)
    return res


@app.route('/deposit/<int:n>', methods=['POST'])
def deposit(n):
    try:
        client = razorpay.Client(auth=(rzp_key, secret_key))
        order = client.order.create({
            "amount": n * 10,  # Amount should be in paise (integer)
            "currency": "INR",
            "receipt": "receipt#1",
            "partial_payment": False,
            "notes": {
                "key1": "value1",
                "key2": "value2"
            }
        })
        return jsonify({"order_id": order['id'], "amount": order['amount']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# The following line should be removed for Vercel deployment
# app.run()

if __name__ == "__main__":
    app.run(debug=True)
