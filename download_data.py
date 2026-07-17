import os
import tarfile
import urllib.request

DATA_DIR = "data"
URL = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
ARCHIVE_PATH = os.path.join(DATA_DIR, "speech_commands.tar.gz")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created {DATA_DIR} directory.")

print("Downloading Google Speech Commands V1 (approx. 1.5GB compressed)...")
urllib.request.urlretrieve(URL, ARCHIVE_PATH)
print("Download complete.")

print("Extracting audio files...")
with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
    tar.extractall(path=DATA_DIR)

os.remove(ARCHIVE_PATH)
print(f"Dataset ready and extracted inside the './{DATA_DIR}' folder!")