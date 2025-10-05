import os
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from sklearn.preprocessing import StandardScaler

class EmotionAudioDataset(Dataset):
    def __init__(self, root_dirs, sr=16000, max_len=3):
        self.sr = sr
        self.max_len = max_len
        self.files = []

        # RAVDESS emotion mapping
        self.ravdess_emotions = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }

        # CREMA-D emotion mapping
        self.cremad_emotions = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad"
        }

        for root in root_dirs:
            self.files.extend(glob(os.path.join(root, "**", "*.wav"), recursive=True))

        all_emotions = sorted(set(self.ravdess_emotions.values()) | set(self.cremad_emotions.values()))
        self.label2id = {emo: idx for idx, emo in enumerate(all_emotions)}
        self.id2label = {idx: emo for emo, idx in self.label2id.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        fname = os.path.basename(path)

        # --- Label extraction ---
        if "Actor_" in path:
            emo_code = fname.split("-")[2]
            emo = self.ravdess_emotions.get(emo_code, "unknown")
        else:
            emo_code = fname.split("_")[2]
            emo = self.cremad_emotions.get(emo_code, "unknown")

        label = self.label2id[emo]

        # --- Load audio ---
        wav, sr = librosa.load(path, sr=self.sr, mono=True)
        num_samples = self.sr * self.max_len
        if len(wav) > num_samples:
            wav = wav[:num_samples]
        else:
            wav = np.pad(wav, (0, num_samples - len(wav)))

        # --- Mel Spectrogram ---
        mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=40)
        mel_db = librosa.power_to_db(mel).T

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=wav, sr=self.sr, n_mfcc=40).T
        mfcc = librosa.util.fix_length(mfcc, size=mel_db.shape[0], axis=0)

        # --- Combine ---
        features = np.concatenate([mel_db, mfcc], axis=1)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return torch.tensor(features, dtype=torch.float32), label


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, num_classes=8, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
