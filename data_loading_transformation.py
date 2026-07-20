import os
import torch
import torchaudio
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
        # Audio Configuration Constants
        self.sample_rate = 16000
        self.n_fft = 512       
        self.hop_length = 320   
        self.n_mels = 32       
        
        # Instantiate transforms once
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # --- THE PATH STORAGE LOOP ---
        # We only loop to collect FILE PATH STRINGS
        self.file_paths = []
        self.labels = []
        
        # Labelling classes
        self.class_to_idx = {"yes": 0, "no": 1, "stop": 2, "go": 3, "unknown": 4}
        
        # skim through target folder
        for category in os.listdir(base_dir):
            category_path = os.path.join(base_dir, category)
            if os.path.isdir(category_path):
                # Map folder name to index (use 'unknown' index if folder isn't in our core 4)
                label_idx = self.class_to_idx.get(category, self.class_to_idx["unknown"])
                
                for filename in os.listdir(category_path):
                    if filename.endswith('.wav'):
                        full_path = os.path.join(category_path, filename)
                        self.file_paths.append(full_path)
                        self.labels.append(label_idx)

    def pad_waveform(self, waveform, target_length=16000):
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_length]
        return waveform

    def __len__(self):
        # The total number of files found
        return len(self.file_paths)

    def __getitem__(self, idx):
        # --- LAZY LOADING HAPPENS HERE ---
        # 1. Grab the path string for this index
        current_path = self.file_paths[idx]
        current_label = self.labels[idx]
        
        # 2. Load the actual audio file into RAM only when asked
        waveform, sr = torchaudio.load(current_path)
        
        # 3. Apply your target dimensions (1, 32, 51)
        waveform = self.pad_waveform(waveform)
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.db_transform(mel_spec)
        
        # 4. Return the processed tensor and its corresponding category label
        return mel_spec_db, current_label




 

