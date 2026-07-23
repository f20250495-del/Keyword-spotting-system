import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import os

# ── CONFIG (config with training data) ─────────────────────
SAMPLE_RATE   = 16000
N_FFT         = 512
HOP_LENGTH    = 320
N_MELS        = 32
MAX_LENGTH    = 16000  # 1 second
CLASS_LABELS  = ['go', 'no', 'stop', 'unknown', 'yes']
MODEL_PATH    = 'best_kws_model.pth'  # adjust to your saved model path

# ── YOUR MODEL CLASS (Specify model architecture) ──────────────
import torch.nn as nn

class KeywordCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8,  kernel_size=(2,5), stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(2,2), stride=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc    = nn.Linear(240, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ── PREPROCESSING ────────────────────────────────────────────────
def preprocess(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform  = resampler(waveform)

    # Pad or truncate to 1 second
    num_samples = waveform.shape[1]
    if num_samples < MAX_LENGTH:
        pad = MAX_LENGTH - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :MAX_LENGTH]

    # Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    db_transform = T.AmplitudeToDB()

    mel = mel_transform(waveform)
    mel = db_transform(mel)

    # Add batch dimension → (1, 1, 32, 51)
    mel = mel.unsqueeze(0)
    return mel

# ── INFERENCE ────────────────────────────────────────────────────
def predict(wav_path):
    if not os.path.exists(wav_path):
        print(f"Error: File not found — {wav_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = KeywordCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Preprocess
    mel = preprocess(wav_path).to(device)

    # Predict
    with torch.no_grad():
        output     = model(mel)
        probs      = torch.softmax(output, dim=1)
        confidence = probs.max().item()
        pred_idx   = probs.argmax().item()
        pred_label = CLASS_LABELS[pred_idx]

    print(f"\nFile: {wav_path}")
    print(f"Predicted keyword: {pred_label.upper()}")
    print(f"Confidence: {confidence*100:.1f}%")
    print("\nAll class probabilities:")
    for label, prob in zip(CLASS_LABELS, probs[0]):
        bar = '█' * int(prob.item() * 20)
        print(f"  {label:10s} {prob.item()*100:5.1f}%  {bar}")

    return pred_label, confidence

# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    
     sample_wav_path = 'data/seven/004ae714_nohash_0.wav' # Replace with an actual file from your dataset
     predict(sample_wav_path)

print("To run inference, uncomment and modify the `predict` function call with a .wav file path.")
print("Example: `predict('004ae714_nohash_1.wav')`")


