import google.generativeai as genai
import os
import re
from dotenv import load_dotenv



load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)
if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found.")
    exit()