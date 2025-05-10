import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, file_list, labels, sr=16000, mel_binos=80, max_len=1000):
        self.file_list = file_list
        self.labels = labels
        self.sr = sr
        self.mel_binos = mel_binos
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, indexo):
        path = self.file_list[indexo]
        label = self.labels[indexo]
        # get audio
        y, _ = librosa.load(path, sr=self.sr)
        # Mel-spectrogramo
        mel = librosa.feature.melspectrogram(y, sr=self.sr, n_mels=self.mel_binos)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        # Pad/truncato
        if mel.shape[1] < self.max_len:
            pad = self.max_len - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad)), mode='constant')
        else:
            mel = mel[:, :self.max_len]
        return torch.from_numpy(mel).float(), label


def collectPhilos(base_dir):
    real_dir = os.path.join(base_dir, 'RealAudios')
    fake_dir = os.path.join(base_dir, 'FakeAudios')
    paths, labels = [], []
    for root, _, files in os.walk(real_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                paths.append(os.path.join(root, f)); labels.append(0)
    for root, _, files in os.walk(fake_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                paths.append(os.path.join(root, f)); labels.append(1)
    return paths, labels
