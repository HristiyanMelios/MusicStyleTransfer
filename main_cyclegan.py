import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import DataLoader

from lib.config import config_args
from lib.dataset import MelDomainDataset
from lib.model import Generator, Discriminator
from lib.train_cyclegan import train_cyclegan
from lib.audio import mel_to_audio, tensor_to_mel


def main():
    args = config_args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"Using device: {device}")

    # Load dataframe
    csv_path = os.path.join(args.metadata_dir, "mels.csv")
    df = pd.read_csv(csv_path)
    df_A = df[df["genre"] == args.genre_A].reset_index(drop=True)
    df_B = df[df["genre"] == args.genre_B].reset_index(drop=True)

    if len(df_A) == 0 or len(df_B) == 0:
        print(f"No valid audio for genres: {args.genre_A} or {args.genre_B}")

    print(f"Loaded {len(df_A)} files for genre: {args.genre_A}")
    print(f"Loaded {len(df_B)} files for genre: {args.genre_B}")

    ds_A = MelDomainDataset(args, df_A)
    ds_B = MelDomainDataset(args, df_B)

    dl_A = DataLoader(ds_A, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers = 4, persistent_workers=True, pin_memory=True)
    dl_B = DataLoader(ds_B, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers = 4, persistent_workers=True, pin_memory=True)

    gen_AtoB = Generator(in_channels=1, out_channels=1, base_channels=64).to(device)
    gen_BtoA = Generator(in_channels=1, out_channels=1, base_channels=64).to(device)
    disc_A = Discriminator(in_channels=1, base_channels=64).to(device)
    disc_B = Discriminator(in_channels=1, base_channels=64).to(device)

    optim_gen = torch.optim.Adam(
        list(gen_AtoB.parameters()) + list(gen_BtoA.parameters()),
        lr=args.g_lr,
        betas=(0.5, 0.999),
    )
    optim_disc = torch.optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=args.d_lr,
        betas=(0.5, 0.999),
    )
    train_cyclegan(args, dl_A, dl_B, gen_AtoB, gen_BtoA, disc_A, disc_B, optim_gen, optim_gen, device)

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(gen_AtoB.state_dict(), os.path.join(checkpoint_dir, 'gen_AtoB.pt'))
    torch.save(gen_BtoA.state_dict(), os.path.join(checkpoint_dir, 'gen_BtoA.pt'))

    print(f"Saved generator checkpoints at {checkpoint_dir}")

    # ------- Output printing -------
    gen_AtoB.eval()
    gen_BtoA.eval()

    samples_A = df_A.sample(5, random_state=args.seed)
    samples_B = df_B.sample(5, random_state=args.seed)
    audio_dir = os.path.join(args.output_dir, 'cyclegan_audio')
    mel_output = os.path.join(args.output_dir, 'cyclegan_mel')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(mel_output, exist_ok=True)

    # Going from A to B
    print(f"Generating audio from Genre {args.genre_A} to {args.genre_B}")
    for _, row in samples_A.iterrows():
        track_id = int(row['track_id'])
        mel = np.load(row["mel_path"])

        # Normalize to [-1, 1]
        S_norm = (mel + 80.0) / 80.0
        S_norm = np.clip(S_norm, 0.0, 1.0)
        S_norm = 2.0 * S_norm - 1.0

        S_tensor = torch.from_numpy(S_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            fake_B = gen_AtoB(S_tensor)

        fake_B = fake_B.cpu()
        # Unnormalize to [0,1] before passing it
        fake_B = (fake_B + 1.0) / 2.0
        fake_B_mel = tensor_to_mel(fake_B)

        # Save output as mel and audio
        mel_path = os.path.join(mel_output, f"cyclegan_AtoB_{track_id:06d}.npy")
        np.save(mel_path, fake_B_mel)
        print(f"Saved generated mel at {mel_path}:"
              f"\ncyclegan_AtoB_{track_id:06d}.npy")

        fake_wav = mel_to_audio(fake_B_mel, args)
        audio_path = os.path.join(audio_dir, f"cyclegan_AtoB_{track_id:06d}.wav")
        sf.write(audio_path, fake_wav, args.sr)
        print(f"Saved generated audio at {audio_path}:"
              f"\ncyclegan_AtoB_{track_id:06d}.wav")

    # Going from B to A
    print(f"Generating audio from Genre {args.genre_B} to {args.genre_A}")
    for _, row in samples_B.iterrows():
        track_id = int(row['track_id'])
        mel = np.load(row["mel_path"])

        # Normalize to [-1, 1]
        S_norm = (mel + 80.0) / 80.0
        S_norm = np.clip(S_norm, 0.0, 1.0)
        S_norm = 2.0 * S_norm - 1.0

        S_tensor = torch.from_numpy(S_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            fake_A = gen_BtoA(S_tensor)

        fake_A = fake_A.cpu()
        # Unnormalize to [0,1] before passing it
        fake_A = (fake_A + 1.0) / 2.0
        fake_A_mel = tensor_to_mel(fake_A)

        # Save output as mel and audio
        mel_path = os.path.join(mel_output, f"cyclegan_BtoA_{track_id:06d}.npy")
        np.save(mel_path, fake_A_mel)
        print(f"Saved generated mel at {mel_path}:"
              f"\ncyclegan_BtoA_{track_id:06d}.npy")

        fake_wav = mel_to_audio(fake_A_mel, args)
        audio_path = os.path.join(audio_dir, f"cyclegan_BtoA_{track_id:06d}.wav")
        sf.write(audio_path, fake_wav, args.sr)
        print(f"Saved generated audio at {audio_path}:"
              f"\ncyclegan_BtoA_{track_id:06d}.wav")


if __name__ == '__main__':
    main()
