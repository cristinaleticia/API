import requests

url = 'http://localhost:5000/analisar'
files = {'imagem': open('cabelo.jpg', 'rb')}

response = requests.post(url, files=files)
print(response.json())