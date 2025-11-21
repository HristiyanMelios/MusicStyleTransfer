import torch
import torch.nn as nn

from lib.model import FeatureExtractors, gram_matrix


def get_content_loss(F_gen, F_content):
    return nn.functional.mse_loss(F_gen, F_content)


def get_style_loss(gen_features, style_features, style_layers,
                   layer_weights):
    loss = 0.0
    for name in style_layers:
        G_gen = gram_matrix(gen_features[name])
        G_style = gram_matrix(style_features[name])
        loss += layer_weights[name] * nn.functional.mse_loss(G_gen,
                                                             G_style)
    return loss


#   TV loss is meant for smoothing in the spectrogram
def get_tv_loss(x):
    loss_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return loss_h + loss_w


def style_transfer(args, content_mel, style_mel, device):
    """
    Main Style Transfer loop

    Args:
        args (Namespace): Namespace object containing command line arguments
        content_mel (Tensor): Content Mel spectrogram
        style_mel (Tensor): Style Mel spectrogram
        device (torch.device): Device used for computation

    Returns:
        generated_mel (Tensor): Generated Mel spectrogram
    """
    vgg_extractor = FeatureExtractors(device=device).to(device)

    vgg_extractor.eval()

    with torch.no_grad():
        content_features = vgg_extractor(content_mel)
        style_features = vgg_extractor(style_mel)

    generated_mel = content_mel.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([generated_mel], lr=args.lr)

    style_layers = ["relu1_1",
                    "relu2_1",
                    "relu3_1",
                    "relu4_1",
                    "relu5_1"
                    ]
    layer_weights = {name: 1.0 for name in style_layers}

    for step in range(args.num_steps):
        optimizer.zero_grad()

        generated_features = vgg_extractor(generated_mel)

        content_loss = get_content_loss(
            generated_features["relu4_2"],
            content_features["relu4_2"],
        )

        style_loss = get_style_loss(generated_features, style_features,
                                    style_layers, layer_weights)

        tv_loss = get_tv_loss(generated_mel)

        # This is to set a specific weight for the amount of style
        # that should be transferred from the target genre
        total_loss = (args.content_weight * content_loss
                      + args.style_weight * style_loss
                      + args.tv_weight * tv_loss)

        # Backprop so the weights get updated
        total_loss.backward()
        optimizer.step()

        # print every 50 steps
        if step % args.print_steps == 0:
            print(
                f"[{step}/{args.num_steps}] Total Loss: {total_loss.item():.6f} "
                f"| Content Loss: {content_loss.item():.6f} "
                f"| Style Loss: {style_loss.item():.6f}")

    return generated_mel.detach()
