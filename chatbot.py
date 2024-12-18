from flask import Flask, request, render_template, session
from google.cloud import vision, texttospeech, speech, firestore
import requests
import os
import tempfile
import PyPDF2
import hashlib

# Flask App
app = Flask(__name__)
app.secret_key = "secretkey123"  # Required for session handling

# Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "magnetic-clone-445004-f0-4b92ab103bb2.json"
vision_client = vision.ImageAnnotatorClient()
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()
db = firestore.Client()
API_KEY = ""
#GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# Function to generate hash
def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Function to extract text from images
def extract_text_from_image(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")
    return " ".join([text.description for text in response.text_annotations])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to handle Gemini API
def generate_gemini_response(prompt):
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=payload, headers=headers)
    return response.json()['candidates'][0]['content']['parts'][0]['text']

# Function to convert text to speech
def text_to_speech(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="hi-IN", name="hi-IN-Wavenet-D")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    audio_file_path = os.path.join("static", "output.mp3")
    with open(audio_file_path, "wb") as out:
        out.write(response.audio_content)
    return audio_file_path

# Function to convert speech to text
def speech_to_text(audio_path):
    with open(audio_path, "rb") as f:
        audio = speech.RecognitionAudio(content=f.read())
    config = speech.RecognitionConfig(language_code="hi-IN")
    response = stt_client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

# Chat History Handling
def add_to_chat_history(user_input, bot_response):
    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"user": user_input, "bot": bot_response})

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    chat_history = session.get("chat_history", [])
    audio_file = None

    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        extracted_text = ""

        # Handle Audio Upload
        if "audio_input" in request.files and request.files["audio_input"].filename:
            audio = request.files["audio_input"]
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio.save(temp_audio.name)
            extracted_text = speech_to_text(temp_audio.name)

        # Handle Image Upload
        elif "document_image" in request.files and request.files["document_image"].filename:
            image = request.files["document_image"]
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            image.save(temp_image.name)
            extracted_text = extract_text_from_image(temp_image.name)

        # Handle PDF Upload
        elif "document_pdf" in request.files and request.files["document_pdf"].filename:
            pdf = request.files["document_pdf"]
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.save(temp_pdf.name)
            extracted_text = extract_text_from_pdf(temp_pdf.name)

        # Handle Text Input or Audio Transcription
        if user_input or extracted_text:
            query = user_input if user_input else extracted_text
            gemini_response = generate_gemini_response(query)
            hindi_response = generate_gemini_response(f"Translate this to Hindi: {gemini_response}")
            audio_file = text_to_speech(hindi_response)

            # Update Chat History
            add_to_chat_history(query, hindi_response)

    return render_template("index_chatbot.html", chat_history=session.get("chat_history", []), audio_file=audio_file)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
