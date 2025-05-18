from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from summarizer import summarize_text
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = pipeline("summarization", model="t5-small",tokenizer="t5-small")
print("Summarizer pipeline loaded successfully.")

@app.route("/",methods=["GET"])
def index():
    return "Flask Summarizer API is running"


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Convert text to speech
    tts = gTTS(text)
    audio_io = BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)

    return send_file(audio_io, mimetype="audio/mpeg", as_attachment=False, download_name="speech.mp3")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")


    if not text.strip():
        return jsonify({"error": "Text is empty."}), 400
    input_text = "summarize: " + text.strip()

    try:
        # Truncate input if too long for model (e.g., >1024 tokens)
        if len(text.split()) > 1024:
            text = " ".join(text.split()[:1024])

        summary = summarizer(
            input_text, 
            max_length=40, 
            min_length=15, 
            do_sample=False)[0]["summary_text"]
        
        return jsonify({"summary": summary})
    

    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)