from flask import Flask, jsonify, request
import razorpay
from dotenv import load_dotenv
import os

# Uncomment the following line if using nemo_toolkit
import nemo.collections.asr as nemo_asr

# Uncomment the following lines if you need speaker recognition
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

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
    emb = speaker_model.get_embedding("./output_audio.wav")

@app.route('/audiocheck')
def audiocheck():
    speaker_model.verify_speakers("./output_audio.wav","./output_audio20240818131440.wav")

@app.route('/deposit/<int:n>', methods=['POST'])
def deposit(n):
    try:
        client = razorpay.Client(auth=(rzp_key, secret_key))
        order = client.order.create({
            "amount": n * 100,  # Amount should be in paise (integer)
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
