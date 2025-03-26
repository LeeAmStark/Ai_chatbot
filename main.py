import base64
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

def generate(user_input):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash"
    
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
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

if __name__ == "__main__":
    print("Ask me anything (or type 'exit' to quit): ", end="")
    
    while True:
        user_input = input()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate(user_input)
        print(response)

