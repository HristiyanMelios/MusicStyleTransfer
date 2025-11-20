"""
This module holds the preprocessing steps for the audio dataset, such as converting the raw .mp3 files into
Mel spectrograms and saving them onto the hard drive to avoid regeneration.
"""

import os
import librosa
import numpy as np
import soundfile as sf


def load_audio(path, args):
    """
    Loads the audio dataset, then trims or pads the files to the desired length.
    """
    y, sr = librosa.load(path, sr=args.sr, mono=True)

    # Only cutting samples down really matters for the FMA dataset, since
    # each audio clip is 30s long.
    target_length = int(args.sr * args.clip_duration)
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        pad_length = target_length - len(y)
        y = np.pad(y, (0, pad_length), 'constant')

    return y, sr


def audio_to_mel(y, args):
    """
    Takes an audio waveform and converts it to a mel spectrogram using
    built in librosa module functions. Outputs the spectrogram in the
    decibel scale rather than the raw power spectrogram.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_size,
        n_mels=128,
        power=2.0,
    )

    # Convert the actual spectrogram into the decibel scale, so we can
    # later use that to normalize into pytorch tensors [0, 1]
    S_in_db = librosa.power_to_db(S, ref=np.max)
    return S_in_db


def mel_to_audio(S_in_db, args, n_iter=32):
    """
    Takes a mel spectrogram (in dB) and converts it to an approximated audio
    waveform using the Griffin-Lim algorithm
    """
    S = librosa.db_to_power(S_in_db)
    y = librosa.feature.inverse.mel_to_audio(
        S,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_size,
        n_iter=n_iter,
    )
    return y


# def save_audio(path, y, sr):
#     """
#     Saves the audio waveform as a .wav file to the specified path
#     using soundfile, as librosa.output.save_wav was deprecated
#     """
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     sf.write(path, y, sr)
