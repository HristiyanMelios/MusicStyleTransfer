"""
Small script to precompute Mel spectrograms from the FMA small dataset,
so we do not have to compute them again.
"""

import os
from audio.py import load_audio, audio_to_mel

# Load dataset here using os

def compute_mels(dataset_path, args):
    # For loop here going through entire dataset
    y, sr = load_audio(dataset_path, args)

    # S = audio_to_mel(y, args)
    # os.path.join(datasetpath + name of the file)
    # numpy save or librosa save here using filepath