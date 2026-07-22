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

#########################################################################################################

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

###############################################################################################3

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

#########################################################################################################

#Design choices:

*Tensor size: (1, 32, 51)
*We require a lightweight model to classify clips into 5 distinct classes(yes,no,stop,go,unknown)

*since we require minimum no of parameter and are using a relatively smaller data set to train and test(prevent overfitting on lightwt model), we like to keep the pixel dimensions minimum.

*our audioclips have more of tonic components rathar than percussive components.(ex:No,go)
We expect a spectrogram with horizontal bands(stable frequency over time) flanked by vertical bands(corresponding to beginning and ending of words)

*keeping all these factors in mind, i decided to have larger no of time frames(512-hiphop time of 20 ms), with each time frame being relatively short in width(25-32 ms compared to std 40 ms)

*Reason: Avoid smearing of vertical bands 


Data split:Google speechset has data divided into 2 files:Training & validation

*Files are divided in manner such that clips of a speaker are isolated to one file
*This prevents memorisation of person ‘s vocal features:pitch, timbre, loudness,
Rathar finds patterns corresponding to keywords.

-How to ensure uniformity in size of training ,val,testing sets?(light architecture)

#CNN architecture:

5 layer network:small input volume, Simple network is enough to identify std keywords.

**For schematic diag- Refer link provided below
*

Parameter count:
Fc:15*16*5=1200+5=1205
Conv 1::3*2*8=48+8=56
Conv 2: 8*(2*2)*16=512+16=528
Total:1789
->Extremely memory and energy efficient:
Optimised for Edge deployment &Embedded systems


(16, 3, 5)


Design choices:
1.1st conv layer:Images contain mainly horizontal bands flanked by vertical bands.
->Stride kept @ 2-features remain variant along horizontal axis
->Dimension kept min:High resolution required to capture”Transition”.
->Features : no of filters kept low - very simple edges(horizontal ,vertical lines)


2.2nd conv layer:
->Stride kept at1 : Patterns involving horizontal bands vary along length
->others remain same

#Final Results and Critical analysis(Refer kink provided below

**https://docs.google.com/document/d/1MBiS5FOwzN77oCfM_pR-MBXG2Y8l8jNy3n1bXKzWoI4/edit?usp=sharing**(Link to doc containing **CNN arhitecture** &**Final results**)










