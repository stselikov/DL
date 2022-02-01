from esrgan.config import *
from esrgan.utils import VGGLoss, save_checkpoint, load_checkpoint, plot_examples
from esrgan.model import Generator, Discriminator, initialize_weights
from esrgan.dataload import both_transforms, highres_transform, lowres_transform

torch.backends.cudnn.benchmark = True

# Training procedure
def train_fn(loader, disc, gen, opt_gen, opt_disc, psnr_loss_f, pixel_loss_f, content_loss_f, adversarial_loss_f, epoch):
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

    loss_d_per_epoch = [] #
    loss_g_per_epoch = []
    psnr_per_epoch = []

    for _, image in enumerate(loop):                            
        high_res = highres_transform(image[0]).to(DEVICE)      
        low_res = lowres_transform(image[0]).to(DEVICE)        
        
        fake = gen(low_res)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        if DISC_TRAIN:
            # At this stage, the discriminator needs to require a derivative gradient
            for p in disc.parameters():
                p.requires_grad = True

            # Initialize the discriminator optimizer gradient
            opt_disc.zero_grad()

            # Calculate the loss of the discriminator on the high-res image
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            
            # Defining losses as described in original paper
            disc_loss_real = adversarial_loss_f(disc_real - torch.mean(disc_fake), torch.ones_like(disc_real))
            disc_loss_fake = adversarial_loss_f(disc_fake - torch.mean(disc_real), torch.zeros_like(disc_fake))
            
            # Gradient backpropagation
            disc_loss_real.backward(retain_graph=True)
            disc_loss_fake.backward()
            opt_disc.step()

            # Count discriminator total loss
            loss_disc = disc_loss_real + disc_loss_fake
            # End training discriminator    

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # At this stage, the discriminator needs not to require a derivative gradient
        for p in disc.parameters():
            p.requires_grad = False

        # Initialize the generator optimizer gradient
        opt_gen.zero_grad()
        # Calculate the loss of the generator on the super-res image
        if DISC_TRAIN:
            # Calculate the loss of the discriminator on the high-res image
            disc_real = disc(high_res.detach())
            disc_fake = disc(fake)

            # Calculate different parts of generator loss, as described in original paper.
            pixel_loss = pixel_weight * pixel_loss_f(fake, high_res.detach())
            content_loss = content_weight * content_loss_f(fake, high_res.detach())
            # Adversarial loss as described in original paper
            adversarial_loss = adversarial_weight * adversarial_loss_f(disc_fake - torch.mean(disc_real), 
                                torch.ones_like(disc_fake))
            # Count generator total loss
            gen_loss = pixel_loss + content_loss + adversarial_loss
        else:
            # Just Generator is trained, no GAN involved
            pixel_loss = pixel_loss_f(fake, high_res.detach())
            gen_loss = pixel_loss
            loss_disc = torch.zeros(1)

        # Gradient backpropagation
        gen_loss.backward()
        opt_gen.step()

        # End training generator

        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_loss_f(fake, high_res))

        loss_d_per_epoch.append(loss_disc.item()) #
        loss_g_per_epoch.append(gen_loss.item()) #
        psnr_per_epoch.append(psnr.item())

        loop.set_postfix(loss_g=gen_loss.item(), loss_d=loss_disc.item(), psnr=psnr.item())

    # Record losses & scores
    losses_g_e = np.mean(loss_g_per_epoch)
    losses_d_e = np.mean(loss_d_per_epoch)
    psnr_e = np.mean(psnr_per_epoch)

    return losses_g_e, losses_d_e, psnr_e
   

def main():
    # Initialize dataset
    dataset = ImageFolder(os.path.join(WORK_PATH, IMAGE_PATH), transform=both_transforms)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    # Define models and losses
    gen = Generator(in_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=3).to(DEVICE)
    initialize_weights(gen)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    psnr_loss_f = nn.MSELoss().to(DEVICE)
    pixel_loss_f = nn.L1Loss().to(DEVICE)
    content_loss_f = VGGLoss().to(DEVICE)
    adversarial_loss_f = nn.BCEWithLogitsLoss().to(DEVICE)

    gen.train()
    disc.train()

    if LOAD_GEN:
        load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
    if LOAD_DISC:
        load_checkpoint(CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE)

    # Losses & scores
    losses_g = []
    losses_d = []
    psnr = []
    
    for epoch in range(NUM_EPOCHS):
        losses_g_e, losses_d_e, psnr_e = train_fn(loader, 
                disc, gen, opt_gen, opt_disc, psnr_loss_f, pixel_loss_f, content_loss_f, adversarial_loss_f, epoch)
        
        losses_g.append(losses_g_e)
        losses_d.append(losses_d_e)
        psnr.append(psnr_e)

        
        if (epoch+1) % PLOT_FREQUENCY == 0:
            print(f"Plotting samples for epoch {epoch+1}" )
            plot_examples(gen)
            gc.collect()
            torch.cuda.empty_cache()

    
    if SAVE_MODEL:
        save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

    gc.collect()
    torch.cuda.empty_cache()
    plot_examples(gen)

    # Show Losses
    plt.figure(figsize=(15, 6))
    plt.plot(losses_d, '-')
    plt.plot(losses_g, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.show()
    # Show PSNR
    plt.figure(figsize=(15, 6))
    plt.plot(psnr, '-')
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.title('PSNR')
    plt.show()


if __name__ == "__main__":
    main()