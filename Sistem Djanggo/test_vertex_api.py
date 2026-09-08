import os
import requests
from google.oauth2 import service_account
import google.auth.transport.requests

key_path = r"d:\TUGAS AKHIR\Final Sistem\n8n-api-keys-499101-c0270bfd2f80.json"

try:
    print(f"Loading credentials from: {key_path}")
    creds = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Refresh credentials to get access token
    print("Requesting access token...")
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    
    token = creds.token
    print("Access token successfully retrieved.")
    
    project_id = creds.project_id
    # Vertex AI Gemini model endpoint. Let's use us-central1 location
    location = "us-central1"
    model_id = "gemini-1.5-flash"  # atau gemini-2.5-flash jika didukung di region tersebut
    
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Halo, sebutkan nama satu kain ulos Batak."}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 100
        }
    }
    
    print(f"Sending POST request to Vertex AI: {url} ...")
    response = requests.post(url, json=payload, headers=headers, timeout=15)
    print(f"Status Code: {response.status_code}")
    print(f"Raw Response: {response.text}")

except Exception as e:
    print(f"FAILED. Error: {e}")
