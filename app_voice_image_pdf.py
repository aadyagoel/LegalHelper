from flask import Flask, request, render_template
from google.cloud import vision, texttospeech, firestore
import requests
import os
import tempfile
import PyPDF2
import hashlib

# Flask App
app = Flask(__name__)

# Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "magnetic-clone-445004-f0-4b92ab103bb2.json"
vision_client = vision.ImageAnnotatorClient()
tts_client = texttospeech.TextToSpeechClient()
db = firestore.Client()
API_KEY = ""
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Function to generate hash of text
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

# Function to generate response using Gemini
def generate_gemini_response(prompt):
    translated_prompt = f"Translate the following text into Hindi: {prompt}"
    payload = {"contents": [{"parts": [{"text": translated_prompt}]}]}
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

# Function to check and fetch duplicate translations
def check_and_get_translation(hash_value):
    docs = db.collection("documents").where("hash", "==", hash_value).stream()
    for doc in docs:
        data = doc.to_dict()
        return data.get("translation")  # Return stored translation
    return None

# Function to store document and translation in Firestore
def store_in_firestore(text, translation, hash_value):
    db.collection("documents").add({
        "text": text,
        "translation": translation,
        "hash": hash_value
    })

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    response_text = ""
    audio_file = None
    duplicate_message = ""

    if request.method == "POST":
        extracted_text = ""

        # Handle text input
        if request.form.get("user_input"):
            extracted_text = request.form["user_input"]

        # Handle image input
        elif "document_image" in request.files and request.files["document_image"].filename:
            image = request.files["document_image"]
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            image.save(temp_image.name)
            extracted_text = extract_text_from_image(temp_image.name)

        # Handle PDF input
        elif "document_pdf" in request.files and request.files["document_pdf"].filename:
            pdf = request.files["document_pdf"]
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.save(temp_pdf.name)
            extracted_text = extract_text_from_pdf(temp_pdf.name)

        if extracted_text:
            # Generate hash for the extracted text
            text_hash = generate_hash(extracted_text)

            # Check if document already exists
            stored_translation = check_and_get_translation(text_hash)
            if stored_translation:
                response_text = stored_translation  # Use stored translation
                duplicate_message = "This document has already been uploaded."
            else:
                # Generate translation if not already uploaded
                response_text = generate_gemini_response(extracted_text)
                store_in_firestore(extracted_text, response_text, text_hash)
                duplicate_message = "Document processed and uploaded to Firestore."

            # Generate speech for the response
            audio_file = text_to_speech(response_text)

    return render_template("index_voice_image_pdf.html", response_text=response_text, audio_file=audio_file, duplicate_message=duplicate_message)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)  # Ensure static folder exists
    app.run(debug=True)
