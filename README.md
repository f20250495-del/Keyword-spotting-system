# Keyword-spotting-system

# Problem statement:
Classification of spoken commands into categories:yes/no/stop/go/unknown

# Repository Structure:

  Keyword-spotting-system/
* .gitignore          Excludes data/, venv/, and system artifacts
*  download_data.py   Fetches and extracts the Speech Commands dataset
* verification.py     Verifies dataset integrity and file counts
* data/               [LOCAL ONLY] Ignored by Git (Contains extracted audio)

*  architecture strictly separates data acquisition logic from the core pipeline.

* pushing the heavy data/ folder to .gitignore, you are keeping the remote GitHub repository lightweight, fast to clone, and strictly focused on source code.

* role of your two scripts:
* download_data.py handles the automated download and extraction
* verification.py runs a programmatic check over the local audio clips so you never accidentally train a model on broken or missing files.

#LOADING DATASET INTO PYTORCH:
*Store Dataset path in raw string variable
*Declare 2 lists:------>file_paths/labels
-	self.file_paths:Stores filepaths of each audio_file as a string
-	self.lables:Store class-intger to which audio is mapped

*Maps class keyword with integer:
- self.class_to_idx = {"yes": 0, "no": 1, "stop": 2, "go": 3, "unknown": 4}
	
**Loop 1**
*Maps folder name to class_integers using get():
- label_idx = self.class_to_idx.get(category, self.class_to_idx["unknown"])
	
**Loop 2**(2 statements)
*Constructs file path by appending category path with filename
- os.path.join(category_path, filename) 
*append file path and label to self.file_paths & self.labels respectively

#DATA TRANSFORMATION:

*Convert raw audio file(.wav) to tensor ( Dimension[1, 32, 51])
-monochannel, 
-32:Number:No of Mel filters
-51:No of timeframes

Audioprocessing parameters:
-Sampling rate(no of points sampled):16000 per/sec
-n_ffts:512 samples allows for time_frame of length(40 ms)
-hopping length:320 allows for time durations of 20 ms
-n_mels:no of mels_filters

*Converting sampled audio-mel_spectrogram:
self.mel_transform = torchaudio.transforms.MelSpectrogram( sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels ) 

*Applying warping function to convert amplitude to loudness
self.db_transform = torchaudio.transforms.AmplitudeToDB() 

*Define padding function:

-Pad sampled  numpy array with zeroes(if short)
-Truncate if longer than 1s.







