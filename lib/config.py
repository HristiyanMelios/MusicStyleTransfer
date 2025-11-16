import argparse

config_args = argparse.ArgumentParser()

#----------------------------------- Dataset Configs -----------------------------------#
config_args.add_argument('--dataset_dir', type=str, default="./data/", help="root directory of dataset")

#----------------------------------- Audio Configs -----------------------------------#
#config_args.add_argument('--sample_rate', type=int, default='something')

#----------------------------------- Preprocessing Configs -----------------------------------#

#----------------------------------- Training Configs -----------------------------------#