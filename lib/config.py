import argparse

config_args = argparse.ArgumentParser()

#----------------------------------- Dataset Configs -----------------------------------#
config_args.add_argument('--dataset_dir', type=str, default="./data/fma_small", help="root directory of dataset")
config_args.add_argument('--metadata_dir', type=str, default="./data/fma_metadata", help="root directory of metadata")
config_args.add_argument('--mel_dir', type=str, default="./data/mel", help="root directory of mel spectrogram")

#----------------------------------- Audio Configs -----------------------------------#
config_args.add_argument('--sr', type=int, default=22050, help="sample rate")
config_args.add_argument('--clip_duration', type=float, default=8.0, help="length of audio sample in seconds")
config_args.add_argument('--hop_size', type=int, default=256, help="hop size")
config_args.add_argument('--n_fft', type=int, default=1024, help="FFT size")

#----------------------------------- Preprocessing Configs -----------------------------------#

#----------------------------------- Training Configs -----------------------------------#
config_args.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
config_args.add_argument('--seed', type=int, default=42)
config_args.add_argument('--lr', type=float, default=1e-4, help="learning rate")
config_args.add_argument('--num_steps', type=int, default=500, help="number of transfer iterations")
config_args.add_argument('--genre_A', type=str, default="Classical")
config_args.add_argument('--genre_B', type=str, default="Jazz")
