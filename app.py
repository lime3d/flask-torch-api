from flask import Flask, request, jsonify #Flask — framework do budowy serwerów www / request — do obsługi zapytań HTTP / jsonify — do wysyłania odpowiedzi w formacie JSON
from flask_cors import CORS #CORS — pozwala na dostęp do serwera z różnych domen (np. z twojego frontendowego localhosta)
import torch
from torchvision import models, transforms
from PIL import Image #PIL.Image — do wczytywania obrazów
import io #io — do obsługi strumieni bajtów (potrzebne do wczytania pliku obrazka przesłanego przez API)
import os
import urllib.request

app = Flask(__name__)
CORS(app)  # pozwala na żądania z innych domen (np. localhost:3000)

# Klasy i model - takie same jak wcześniej
classes = ['mak_polny', 'miotla_zbozowa', 'zdrowe']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = 'https://uprawy.faiqkoahkd.cfolks.pl/model.pth'  # <-- podmień na swój URL
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pth')

if not os.path.exists(MODEL_PATH):
    print("Pobieram model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model pobrany.")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = classes[predicted.item()]
        return jsonify({'class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    print("Ping endpoint hit!")
    return jsonify({"message": "pong"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
