import argparse

config_args = argparse.ArgumentParser()

#----------------------------------- Dataset Configs -----------------------------------#
config_args.add_argument('--dataset_dir', type=str, default="./data/fma_small", help="Root directory of dataset")
config_args.add_argument('--metadata_dir', type=str, default="./data/fma_metadata", help="Root directory of metadata")
config_args.add_argument('--mel_dir', type=str, default="./data/mel", help="Root directory of mel spectrogram")
config_args.add_argument('--output_dir', type=str, default="./outputs", help="Output directory")

#----------------------------------- Audio Configs -----------------------------------#
config_args.add_argument('--sr', type=int, default=22050, help="Sample rate")
config_args.add_argument('--clip_duration', type=float, default=8.0, help="Length of audio sample in seconds")
config_args.add_argument('--hop_size', type=int, default=256, help="Hop size")
config_args.add_argument('--n_fft', type=int, default=1024, help="FFT size")

#----------------------------------- Training Configs -----------------------------------#
config_args.add_argument('--device_name', type=str, default="0", help="GPU name")
config_args.add_argument('--seed', type=int, default=42, help="Random seed for reproducability")
config_args.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
config_args.add_argument('--genre_A', type=str, default="Pop", help="Initial audio genre")
config_args.add_argument('--genre_B', type=str, default="Rock", help="Transfer audio genre")

#----------------------------------- CNN Configs -----------------------------------#
config_args.add_argument('--num_steps', type=int, default=500, help="Number of transfer iterations")
config_args.add_argument('--content_weight', type=float, default=1, help="Content weight")
config_args.add_argument('--style_weight',type=float, default=1e5, help="Weight of style transfer")
config_args.add_argument('--tv_weight', type=float, default=1e-6, help="Total variation regularization weight")
config_args.add_argument('--print_steps', type=int, default=50, help="Print progress during training")

#----------------------------------- CycleGAN Configs -----------------------------------#
config_args.add_argument('--batch_size', type=int, default=16, help="Batch size")
config_args.add_argument('--num_epochs', type=int, default=50, help="Number of epochs")
config_args.add_argument('--g_lr', type=float, default=1e-4, help="Generator learning rate")
config_args.add_argument('--d_lr', type=float, default=1e-4, help="Discriminator learning rate")
config_args.add_argument('--cycle_weight', type=float, default=10.0, help="Cycle weight")
config_args.add_argument('--identity_weight', type=float, default=5.0, help="Identity Loss weight")
config_args.add_argument('--patch_time', type=int, default=256, help="Spectrogram patch time frames")
