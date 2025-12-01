import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_mel(spectrogram, title, outfile, vmin=-80, vmax=0):
    plt.figure(figsize=(10, 4))
    im = plt.imshow(spectrogram, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im, format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time frames')
    plt.ylabel('Mel bins')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def cnn_plots(args):
    print("Creating cnn plots...")
    mel_dir = "./outputs/mel"
    out_dir = "./outputs/mel_plots"
    os.makedirs(out_dir, exist_ok=True)

    files = os.listdir(mel_dir)
    generated = sorted(f for f in files if
                       f.startswith("generated") and f.endswith(".npy"))

    if not generated:
        print("No generated files found in the mel directory")
        return

    style = sorted(f for f in files if
                   f.startswith("style") and f.endswith(".npy"))
    if not style:
        print("No style files found in the mel directory")
        return
    style_path = os.path.join(mel_dir, style[0])

    for gen_file in generated:
        content_id = gen_file[len("generated_"):-4]
        generated_path = os.path.join(mel_dir, gen_file)
        content_path = os.path.join(mel_dir, f"content_{content_id}.npy")

        if not os.path.exists(content_path):
            print(f"No content file found, skipping {gen_file}")
            continue

        content_mel = np.load(content_path)
        generated_mel = np.load(generated_path)

        # plot
        content_plot = os.path.join(out_dir, f"content_{content_id}.png")
        plot_mel(content_mel, f"Content {content_id}", content_plot)

        style_id = os.path.basename(style_path)[len("style_"):-4]
        style_mel = np.load(style_path)
        style_plot = os.path.join(out_dir, f"style_{style_id}_content_{content_id}.png")
        plot_mel(style_mel, f"Style {style_id}", style_plot)

        generated_plot = os.path.join(out_dir, f"generated_{content_id}.png")
        plot_mel(generated_mel, f"Generated {content_id}", generated_plot)

        # Numerical difference
        diff_mel = content_mel - generated_mel
        diff_plot = os.path.join(out_dir, f"diff_{content_id}.png")
        plot_mel(diff_mel, f"Difference {content_id}", diff_plot, vmin=-10, vmax=0)
        diff_cg = np.mean(np.abs(content_mel - generated_mel))
        diff_sg = np.mean(np.abs(style_mel - generated_mel))
        print(f"Mean | content - generated |: {diff_cg:.4f} dB")
        print(f"Mean | style - generated |: {diff_sg:.4f} dB")


def cyclegan_plots(args):
    print("Creating cyclegan plots...")
    mel_dir = "./outputs/cyclegan_mel"
    out_dir = "./outputs/cyclegan_plots"
    mel_data_dir = "./data/mel"
    os.makedirs(out_dir, exist_ok=True)

    # CycleGAN is not paired, so it is a lot more complicated to
    # format the filenames compared to just copying cnn implementation
    files = sorted(f for f in os.listdir(mel_dir) if
                   f.startswith("cyclegan_") and f.endswith(".npy"))

    if not files:
        print("No cyclegan files found in the mel directory")
        return

    for file in files:
        # Split into components since output is like cyclegan_AtoB_trackid.npy
        name_parts = file[:-4].split("_")
        if len(name_parts) != 3:
            print(f"unexpected name format: {file}")
            continue

        _, direction, track_id = name_parts

        gen_mel = os.path.join(mel_dir, file)

        subdir = track_id[:3]
        original_path = os.path.join(mel_data_dir, subdir, f"{track_id}.npy")

        if not os.path.exists(original_path):
            print(f"No original mel file found, skipping {file}")
            continue

        original_mel = np.load(original_path)
        generated_mel = np.load(gen_mel)

        # Plots
        original_plot = os.path.join(out_dir, f"original_{track_id}.png")
        plot_mel(original_mel, f"Original {track_id}", original_plot)

        generated_plot = os.path.join(out_dir, f"generated_{direction}_{track_id}.png")
        plot_mel(generated_mel, f"Generated {direction}_{track_id}", generated_plot)

        # Plot the difference since it's very hard to notice
        diff_mel = original_mel - generated_mel
        diff_plot = os.path.join(out_dir, f"diff_{direction}_{track_id}.png")
        plot_mel(diff_mel, f"Difference {direction}_{track_id}", diff_plot, vmin=-10, vmax=0)

        # Numerical difference
        diff = np.mean(np.abs(original_mel - generated_mel))
        print(f"Mean {direction} {track_id}| original - generated |: {diff:.4f} dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="cnn",
        choices=["cnn", "cyclegan"],
        help="Which model to run outputs for",
    )
    args = parser.parse_args()
    if args.mode == "cnn":
        cnn_plots(args)
    elif args.mode == "cyclegan":
        cyclegan_plots(args)


if __name__ == "__main__":
    main()
