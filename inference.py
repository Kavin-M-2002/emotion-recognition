import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import librosa
import numpy as np
from train_emotion_audio import EmotionAudioDataset, BiLSTMClassifier

app = FastAPI(title="Speech Emotion Recognition")

# Load dataset metadata (for label mapping)
dataset = EmotionAudioDataset(["./ravdess_data", "./cremad_data/AudioWAV"])
num_classes = len(dataset.label2id)
input_dim = 80  # must match training config

# Load trained model
model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=128, num_classes=num_classes)
model.load_state_dict(torch.load("bilstm_emotion_v2.pt", map_location="cpu"))
model.eval()

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def preprocess(path):
    sr = 16000
    wav, _ = librosa.load(path, sr=sr, mono=True)
    num_samples = sr * 3
    if len(wav) > num_samples:
        wav = wav[:num_samples]
    else:
        wav = np.pad(wav, (0, num_samples - len(wav)))

    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel).T

    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40).T
    mfcc = librosa.util.fix_length(mfcc, size=mel_db.shape[0], axis=0)

    features = np.concatenate([mel_db, mfcc], axis=1)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return x

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    with open("temp.wav", "wb") as f:
        f.write(await file.read())

    x = preprocess("temp.wav")

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
        confidence = float(probs[pred] * 100)
        emotion = dataset.id2label[pred]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "emotion": emotion,
            "confidence": f"{confidence:.2f}"
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
