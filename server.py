from flask import Flask, jsonify, request
import razorpay
from dotenv import load_dotenv
import os
import nemo.collections.asr as nemo_asr

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")


load_dotenv()
app = Flask(__name__)

rzp_key = os.getenv('RAZORPAY_KEY')
secret_key = os.getenv('RAZORPAY_SECRET_KEY')

@app.route('/members')
def members():
    return jsonify({"name": "John"})

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


app.run()
