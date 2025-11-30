import torch
import torch.nn as nn
import torch.nn.functional as F


def gan_loss(pred, is_real, args):
    # Choose loss type between MSE or original loss
    if is_real:
        target = torch.ones_like(pred)
    else:
        target = torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def train_cyclegan(args, dl_A, dl_B, generator_AtoB, generator_BtoA, discriminator_A, discriminator_B, optimizer_gen, optimizer_disc, device):
    l1_loss = nn.L1Loss()

    # Training Loop
    for epoch in range(args.num_epochs):
        generator_AtoB.train()
        generator_BtoA.train()
        discriminator_A.train()
        discriminator_B.train()

        for step, (batch_A, batch_B) in enumerate(zip(dl_A, dl_B)):
            real_A = batch_A.to(device)
            real_B = batch_B.to(device)

            optimizer_gen.zero_grad()

            # Converting A to B and back to A
            fake_B = generator_AtoB(real_A)
            recovered_A = generator_BtoA(fake_B)

            # B to A and back to B
            fake_A = generator_BtoA(real_B)
            recovered_B = generator_AtoB(fake_A)

            # GAN Loss
            gan_loss_AtoB = gan_loss(discriminator_B(fake_B), True, args)
            gan_loss_BtoA = gan_loss(discriminator_A(fake_A), True, args)

            # Cycle Loss
            cycle_loss_A = l1_loss(recovered_A, real_A)
            cycle_loss_B = l1_loss(recovered_B, real_B)
            total_cycle_loss = cycle_loss_A + cycle_loss_B

            # Identity Loss
            identity_A = generator_BtoA(real_A)
            identity_B = generator_AtoB(real_B)

            identity_loss_A = l1_loss(identity_A, real_A)
            identity_loss_B = l1_loss(identity_B, real_B)
            total_identity_loss = identity_loss_A + identity_loss_B

            generator_loss = (
                gan_loss_AtoB + gan_loss_BtoA
                + args.cycle_weight * total_cycle_loss
                + args.identity_weight * total_identity_loss
            )
            generator_loss.backward()
            optimizer_gen.step()

            # Discriminator training
            optimizer_disc.zero_grad()

            # Discriminator A
            discriminator_loss_realA = gan_loss(discriminator_A(real_A), True, args)
            discriminator_loss_fakeA = gan_loss(discriminator_A(fake_A.detach()), False, args)
            discriminator_lossA = 0.5 * (discriminator_loss_realA + discriminator_loss_fakeA)

            # Discriminator B
            discriminator_loss_realB = gan_loss(discriminator_B(real_B), True, args)
            discriminator_loss_fakeB = gan_loss(discriminator_B(fake_B.detach()), False, args)
            discriminator_lossB = 0.5 * (discriminator_loss_realB + discriminator_loss_fakeB)

            discriminator_loss = discriminator_lossA + discriminator_lossB
            discriminator_loss.backward()
            optimizer_disc.step()

            if step % args.print_steps == 0:
                print(
                    f"[Epoch {epoch}/{args.num_epochs}] "
                    f"[Step {step}] "
                    f"Generator Loss: {generator_loss.item():.4f} "
                    f"Discriminator Loss: {discriminator_loss.item():.4f} "
                    f"Cycle Loss: {total_cycle_loss.item():.4f} "
                    f"Identity Loss: {total_identity_loss.item():.4f} "
                )
    return generator_AtoB, generator_BtoA, discriminator_A, discriminator_B
