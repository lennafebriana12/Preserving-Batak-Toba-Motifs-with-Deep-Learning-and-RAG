import os
import google.generativeai as genai
from google.oauth2 import service_account

key_path = r"d:\TUGAS AKHIR\Final Sistem\n8n-api-keys-499101-c0270bfd2f80.json"

try:
    print(f"Loading credentials from: {key_path}")
    # Kami meload service account credentials
    creds = service_account.Credentials.from_service_account_file(key_path)
    
    # Configure genai dengan credentials
    print("Configuring google-generativeai with credentials...")
    genai.configure(credentials=creds)
    
    # Coba generate content menggunakan model
    # Catatan: Karena menggunakan service account (biasanya terikat ke GCP Vertex AI / Google Cloud),
    # mari kita lihat apakah model-model Gemini di google-generativeai developer API mendukung service account ini
    print("Attempting to call gemini-2.5-flash...")
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("test")
    print(f"SUCCESS! Response: {response.text.strip()}")

except Exception as e:
    print(f"FAILED. Error: {e}")
