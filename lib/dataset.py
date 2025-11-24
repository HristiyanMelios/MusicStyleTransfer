"""
    This module creates a custom PyTorch Dataset for loading mel spectrogram numpy files from a CSV DataFrame.

    Each row in the DataFrame contains track_id, genre, audio_path, and mel_path.

"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_mel_metadata(args):
    csv_path = os.path.join(args.metadata_dir, "mels.csv")
    df = pd.read_csv(csv_path)
    return df


class CSVDataset(Dataset):
    """
    Initialize the dataset with arguments and data.
    Initialize the image and mask transformation pipeline for pre-processing.

    Args:
        args: Configuration ArgumentParser.
        df (pd.DataFrame): DataFrame containing:
         'track_id', 'genre', 'audio_path', 'mel_path' columns.
    """
    def __init__(self, args, df: pd.DataFrame):
        self.args = args
        self.df: pd.DataFrame = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        spectrogram = np.load(row["mel_path"])

        S_norm = (spectrogram + 80.0) / 80.0
        S_norm = np.clip(S_norm, 0.0, 1.0)
        x = torch.from_numpy(S_norm).unsqueeze(0).float()

        if row["genre"] == self.args.genre_A:
            y = torch.tensor(0)
        else:
            y = torch.tensor(1)

        track_id = int(row["track_id"])

        return x, y, track_id


# This dataset is for the CycleGAN implementation
class MelDomainDataset(Dataset):
    def __init__(self, args, df, patch_time=None, transform=None):
        self.args = args
        self.df = df.reset_index(drop=True)
        if patch_time is not None:
            self.patch_time = patch_time
        else:
            self.patch_time = self.args.patch_time
        # If transformations are going to be applied
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        spectrogram = np.load(row["mel_path"])

        S_norm = (spectrogram + 80.0) / 80.0
        S_norm = np.clip(S_norm, 0.0, 1.0)

        n_mels, T = S_norm.shape
        pt = self.patch_time

        if T > pt:
            start = np.random.randint(0, T - pt + 1)
            S_crop = S_norm[:, start:start + pt]
        else:
            pad_width = pt - T
            S_crop = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)

        S_crop = 2.0 * S_crop - 1.0

        x = torch.from_numpy(S_crop).unsqueeze(0).float()

        if self.transform:
            x = self.transform(x)

        return x
