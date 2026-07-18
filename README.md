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



