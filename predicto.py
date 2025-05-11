import os
import torch
import torchaudio
from modelo import VeriVoice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VeriVoice().to(device)
checkpoint = torch.load('modelo.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

def preprocess(path, sr=16000, melBinos=80, max_len=1000):
    waveform, orig_sr = torchaudio.load(path)  # (channels, L)
    waveform = waveform.mean(dim=0, keepdim=True)  # (1, L)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=melBinos
    )
    mel_spec = mel_spec_transform(waveform)       # (1, melBinos, T)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    mel_db = db_transform(mel_spec)              # (1, melBinos, T)
    mel = mel_db[0]                               # (melBinos, T)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6) #normalizo
    # pad/truncato
    T = mel.size(1)
    if T < max_len:
        pad_amt = max_len - T
        mel = torch.nn.functional.pad(mel, (0, pad_amt))
    else:
        mel = mel[:, :max_len]
    return mel  # shape: (melBinos, max_len)

def predicto(path):
    mel = preprocess(path).to(device)            # (melBinos, max_len)
    x = mel.unsqueeze(0)                         # (1, melBinos, max_len)
    with torch.no_grad():
        logits = model(x)                        # (1, numClasses)
        probs = torch.softmax(logits, dim=1)
        predo = probs.argmax(dim=1).item()
        conf = probs[0, predo].item()
    labelo = 'Fake' if predo == 1 else 'Real'
    return labelo, conf

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python3 -m {os.path.basename(__file__)} <audio_path.wav>")
        sys.exit(1)
    audioPhile = sys.argv[1]
    if not os.path.isfile(audioPhile):
        print(f"Error: File '{audioPhile}' not found.")
        sys.exit(1)
    labelo, confidence = predicto(audioPhile)
    print(f"{audioPhile}: {labelo} (Confidence: {confidence*100:.1f}%)")
