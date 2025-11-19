"""
Small script to precompute Mel spectrograms from the FMA small dataset,
so we do not have to compute them again.
"""

import os
import numpy as np
import pandas as pd

from lib.config import config_args
from lib.audio import load_audio, audio_to_mel


def load_metadata(args):
    """
    Loads metadata from the fma_metadata.csv file, and filters to the 'small' dataset,
    selected genre, and builts a DataFrame with track_id, genre, and audio_path
    for us to use later
    """
    track_path = os.path.join(args.metadata_dir, "tracks.csv")
    tracks = pd.read_csv(track_path, index_col=0, header=[0,1])

    # Only keep the tracks in the small subset
    tracks = tracks[tracks[("set", "subset")] == "small"]

    # Only use whichever two genres were passed in as a config
    mask = tracks[("track", "genre_top")].isin([args.genre_A, args.genre_B])
    tracks = tracks[mask]

    df = pd.DataFrame({
        "track_id": tracks.index,
        "genre": tracks[("track", "genre_top")].values,
    })

    # Check on this, might just be able to do it out of a helper method
    def make_audio_path(track_id):
        track_id_str = f"{track_id:06d}"  # Format to 6 digits as FMA does
        return os.path.join(args.dataset_dir, track_id_str[:3], track_id_str + ".mp3")

    df["audio_path"] = df["track_id"].apply(make_audio_path)
    return df


def main():
    args = config_args.parse_args()

    # Create directory if not already made to store Mel spectrograms
    os.makedirs(args.mel_dir, exist_ok=True)

    print("Loading FMA metadata")
    df = load_metadata(args)
    print(f"Found {len(df)} tracks in subset 'small' with genres "
          f"{args.genre_A} and {args.genre_B}")

    mel_paths = []
    masks = []

    for idx, row in df.iterrows():
        track_id = row["track_id"]
        audio_path = row["audio_path"]
        track_id_str = f"{track_id:06d}"

        # Follow FMA folder structure
        mel_subdir = os.path.join(args.mel_dir, track_id_str[:3])
        os.makedirs(mel_subdir, exist_ok=True)
        mel_path = os.path.join(mel_subdir, track_id_str + ".npy")

        try:
            # Load audio
            y, sr = load_audio(audio_path)

            # Convert to Mel Spectrogram
            S_in_db = audio_to_mel(y, args)

            # Save as a .npy for processing
            np.save(mel_path, S_in_db)

            mel_paths.append(mel_path)
            masks.append(True)

            if len(mel_paths) % 100 == 0:
                print(f"Processed {len(mel_paths)} tracks")

        except Exception as e:
            print(f"Error processing track={track_id}, path={audio_path}: {e}")
            mel_paths.append(None)
            masks.append(False)

    # Add all paths and drop any errors
    df["mel_path"] = mel_paths
    df = df[masks]

    # Save to csv so that we can use it later
    out_csv = os.path.join(args.metadata_dir, "mels.csv")
    df.to_csv(out_csv, index=False)
    print(f"Finished processing {len(df)} tracks")
    print(f"Saved metadata to: {out_csv}")


if __name__ == "__main__":
    main()
