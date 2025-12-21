import requests

GRADIO_URL = "https://aaad7868d495b1f819.gradio.live"  # replace this

payload = {
    "data": [
        "Explain post-harvest losses in Nigerian agriculture in simple terms."
    ]
}

response = requests.post(
    f"{GRADIO_URL}/api/predict",
    json=payload,
    timeout=60
)

print(response.json())