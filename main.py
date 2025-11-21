import os

import numpy as np
import torch
import soundfile as sf

from lib.config import config_args
from lib.dataset import load_mel_metadata
from lib.audio import mel_to_audio
from lib.train import style_transfer


def mel_to_tensor(spectrogram):
    S_norm = (spectrogram + 80.0) / 80.0
    S_norm = np.clip(S_norm, 0.0, 1.0)
    # Unsqueeze twice to get the 4D channel we need for transfer
    tensor = torch.from_numpy(S_norm).unsqueeze(0).unsqueeze(0).float()
    return tensor


def tensor_to_mel(tensor):
    S_norm = tensor.squeeze(0).squeeze(0).cpu().numpy()
    S_norm = S_norm * 80.0 - 80.0
    return S_norm


def main():
    # Setting device and seed
    args = config_args.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Creating all the output directories
    audio_output_dir = os.path.join(args.output_dir, 'audio')
    mel_output_dir = os.path.join(args.output_dir, 'mel')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(mel_output_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Load dataframe
    df = load_mel_metadata(args)

    df_A = df[df["genre"] == args.genre_A].reset_index(drop=True)
    df_B = df[df["genre"] == args.genre_B].reset_index(drop=True)

    if len(df_A) == 0 or len(df_B) == 0:
        raise RuntimeError(
            print(f"No valid audio files for one of the genres:"
                  f"{args.genre_A}, {args.genre_B} ")
        )

    print(f"Loaded {len(df_A)} audio files for genre {args.genre_A}")
    print(f"Loaded {len(df_B)} audio files for genre {args.genre_B}")

    content_row = df_A.sample(1, random_state=args.seed).iloc[0]
    style_row = df_B.sample(1, random_state=args.seed).iloc[0]

    print(f"Selected content track: {content_row}")
    print(f"Selected style track: {style_row}")

    # Load mel spectrograms, convert to tensor, run transfer
    content_mel = np.load(content_row["mel_path"])
    style_mel = np.load(style_row["mel_path"])

    content_tensor = mel_to_tensor(content_mel).to(device)
    style_tensor = mel_to_tensor(style_mel).to(device)

    print(f"Running style transfer...")
    generated_tensor = style_transfer(args, content_tensor, style_tensor, device)

    content_id = int(content_row["track_id"])
    style_id = int(style_row["track_id"])

    content_mel_path = os.path.join(mel_output_dir, f"content_{content_id:06d}.npy")
    style_mel_path = os.path.join(mel_output_dir, f"style_{style_id:06d}.npy")
    generated_mel_path = os.path.join(mel_output_dir, f"generated_{content_id:06d}.npy")

    np.save(content_mel_path, content_mel)
    np.save(style_mel_path, style_mel)
    np.save(generated_mel_path, tensor_to_mel(generated_tensor))

    # Audio conversion and saving
    content_audio = mel_to_audio(content_mel, args)
    style_audio = mel_to_audio(style_mel, args)
    generated_mel = tensor_to_mel(generated_tensor)
    generated_audio = mel_to_audio(generated_mel, args)

    content_wav_path = os.path.join(audio_output_dir, f"content_{content_id:06d}.wav")
    style_wav_path = os.path.join(audio_output_dir, f"style_{style_id:06d}.wav")
    generated_wav_path = os.path.join(audio_output_dir, f"generated_{content_id:06d}.wav")

    sf.write(content_wav_path, content_audio, args.sr)
    sf.write(style_wav_path, style_audio, args.sr)
    sf.write(generated_wav_path, generated_audio, args.sr)

    print(f"Generated audio files:"
          f"\n{content_wav_path}"
          f"\n{style_wav_path}"
          f"\n{generated_wav_path}")

    print("Finished transfer.")


if __name__ == "__main__":
    main()
