import requests

url = "https://orleans-arc-port-constraints.trycloudflare.com/generate"
payload = {
  "prompt": "explain wireless network in 5G",
  "max_tokens": 256
}

r = requests.post(url, json=payload, timeout=60)
print(r.json())
