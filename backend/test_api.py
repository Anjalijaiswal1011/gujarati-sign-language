import requests
import json

url = "http://127.0.0.1:8000/api/translator/voice/text-to-speech/"
payload = {"text": "નમસ્તે", "language": "gu"}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    print("Status:", response.status_code)
    print("Response:", response.text[:500])  # Print first 500 chars to avoid huge base64 spam
except Exception as e:
    print("Error:", e)
