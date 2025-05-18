from flask import Flask, jsonify, request
import razorpay
from dotenv import load_dotenv
import os
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist
import firebase_admin
from firebase_admin import credentials, storage
import subprocess
import tempfile
import uuid
from urllib.parse import unquote, urlparse

# Load the model and create embeddings
model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")

load_dotenv()
app = Flask(__name__)

# Retrieve Razorpay credentials from environment variables
rzp_key = os.getenv('RAZORPAY_KEY')
secret_key = os.getenv('RAZORPAY_SECRET_KEY')

# print(os.getenv('FIREBASE_STORAGE_BUCKET'))

# Initialize Firebase
firebase_cred = credentials.Certificate({
    "type": "service_account",
    "project_id": "voice-pay-ac033",
    "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
    "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
    "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
    "client_id": os.getenv('FIREBASE_CLIENT_ID'),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL'),
    "universe_domain": "googleapis.com"
})

firebase_app = firebase_admin.initialize_app(firebase_cred, {
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
})
bucket = storage.bucket()

# Helper function to extract path from Firebase URL
def extract_firebase_path(url):
    if url.startswith('https://firebasestorage.googleapis.com'):
        parsed = urlparse(url)
        path = unquote(parsed.path.split('/o/')[1].split('?')[0])
        return path.strip('/')
    return url.strip('/')

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/members')
def members():
    return jsonify({"name": "John"})

# @app.route('/audiocheck')
# def audiocheck():
#     embedding1 = inference("../../demoaudio/output2.wav")
#     embedding2 = inference("../../demoaudio/output3.wav")

#     # Ensure embeddings are 2D (reshaped if necessary)
#     embedding1 = embedding1.reshape(1, -1)
#     embedding2 = embedding2.reshape(1, -1)

#     # Calculate cosine distance between the two embeddings
#     distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]

#     print(f"Cosine distance between the embeddings: {distance}")
#     return str(distance)

@app.route('/audiocheck', methods=['POST'])
def audiocheck():
    try:
        data = request.get_json()
        file_url1 = data.get('file_url1')
        file_url2 = data.get('file_url2')
        
        if not file_url1 or not file_url2:
            return jsonify({"error": "Both file_url1 and file_url2 are required"}), 400
        
        # Get proper storage paths
        file_path1 = extract_firebase_path(file_url1)
        file_path2 = extract_firebase_path(file_url2)
        
        # Download files temporarily
        temp_dir = tempfile.gettempdir()
        local_path1 = os.path.join(temp_dir, f"temp1_{uuid.uuid4()}.wav")
        local_path2 = os.path.join(temp_dir, f"temp2_{uuid.uuid4()}.wav")
        
        bucket.blob(file_path1).download_to_filename(local_path1)
        bucket.blob(file_path2).download_to_filename(local_path2)
        
        # Process audio files
        embedding1 = inference(local_path1).reshape(1, -1)
        embedding2 = inference(local_path2).reshape(1, -1)
        distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]
        
        # Cleanup
        os.remove(local_path1)
        os.remove(local_path2)
        
        return jsonify({
            "similarity_score": float(1 - distance),  # Convert distance to similarity
            "message": "Audio comparison successful"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        data = request.get_json()
        file_url = data.get('file_url')
        username = data.get('username')
        
        if not file_url or not username:
            return jsonify({"error": "Both file_url and username are required"}), 400
        
        # Get proper storage path
        file_path = extract_firebase_path(file_url)
        
        # Download file temporarily
        temp_dir = tempfile.gettempdir()
        original_path = os.path.join(temp_dir, f"original_{uuid.uuid4()}.wav")
        processed_path = os.path.join(temp_dir, f"processed_{username}.wav")
        
        bucket.blob(file_path).download_to_filename(original_path)
        
        # Process with ffmpeg to set exact metadata and audio specifications
        subprocess.run([
            'ffmpeg',
            '-i', original_path,                   # Input file
            '-ac', '1',                           # 1 channel (mono)
            '-ar', '16000',                       # 16kHz sample rate
            '-c:a', 'pcm_s16le',                  # 16-bit PCM encoding
            '-metadata', f'Software=Lavf61.9.100', # Software metadata
            '-metadata', f'artist={username}',     # Custom artist metadata
            '-metadata', f'title=Processed by {username}',
            '-y',                                  # Overwrite output file if exists
            processed_path
        ], check=True)
        
        # Verify the processed file meets specifications
        verify_audio_specs(processed_path)
        
        # Upload processed file
        processed_blob_name = f"processed_audio/{username}/{username}.wav"
        processed_blob = bucket.blob(processed_blob_name)
        processed_blob.upload_from_filename(processed_path)
        processed_blob.make_public()
        
        # Cleanup
        os.remove(original_path)
        os.remove(processed_path)
        
        return jsonify({
            "processed_file_url": processed_blob.public_url,
            "message": "Audio processed successfully with exact specifications"
        })
        
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg processing failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def verify_audio_specs(file_path):
    """Verify the output file meets our required specifications"""
    result = subprocess.run([
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=channels,sample_rate,codec_name,bits_per_sample',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1',
        file_path
    ], capture_output=True, text=True)
    
    # Parse ffprobe output
    specs = {}
    for line in result.stdout.split('\n'):
        if '=' in line:
            key, value = line.split('=')
            specs[key] = value.strip()
    
    # Verify specifications
    if not (specs.get('channels') == '1' and
            specs.get('sample_rate') == '16000' and
            specs.get('codec_name') == 'pcm_s16le' and
            specs.get('bits_per_sample') == '16'):
        raise ValueError("Output file doesn't meet required specifications")
    
    
if __name__ == "__main__":
    app.run(debug=True)
