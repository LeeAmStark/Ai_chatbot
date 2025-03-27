from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allows frontend requests

def generate(user_input):
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = "gemini-2.0-flash"
    
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)]),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_MEDIUM_AND_ABOVE"),
        ],
        response_mime_type="text/plain",
    )

    response = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        response += chunk.text
    return response

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    try:
        ai_response = generate(data["message"])
        return jsonify({"response": ai_response})
    except Exception as e:
        print("❌ AI Generation Error:", str(e))  # Logs in terminal
        return jsonify({"error": "AI failed to generate a response"}), 500

if __name__ == "__main__":
    app.run(debug=True)  # Runs on http://127.0.0.1:5000
