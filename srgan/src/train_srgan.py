from srgan.config import *
from srgan.utils import VGGLoss, save_checkpoint, load_checkpoint, plot_examples
from srgan.dataload import MyImageFolder
from srgan.model import Generator, Discriminator

torch.backends.cudnn.benchmark = True

# Training procedure
def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, epoch):
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

    loss_d_per_epoch = [] 
    loss_g_per_epoch = []
    real_score_per_epoch = []
    fake_score_per_epoch = []

    for _, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(DEVICE)
        low_res = low_res.to(DEVICE)
        
        fake = gen(low_res)
        l2_loss = mse(fake, high_res)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        if DISC_TRAIN:
            disc_real = disc(high_res)
            cur_real_score = torch.mean(torch.sigmoid(disc_real)).item() #
            
            disc_fake = disc(fake.detach())
            cur_fake_score = torch.mean(torch.sigmoid(disc_fake)).item() #
        
            disc_loss_real = bce(disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)) # Label smoothing
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake)) 
            loss_disc = disc_loss_fake + disc_loss_real

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            disc_fake = disc(fake)

            # Define different losses for Generattor
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = (1.0/12.75) * vgg_loss(fake, high_res)
            gen_loss = l2_loss + loss_for_vgg + adversarial_loss
        else:
            gen_loss = l2_loss
            # Define zero losses and scores if not training the Discriminator
            # just for uniform output representation
            cur_real_score = 0
            cur_fake_score = 0
            loss_disc = torch.zeros(1)

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        loss_d_per_epoch.append(loss_disc.item()) 
        loss_g_per_epoch.append(gen_loss.item()) 
        real_score_per_epoch.append(cur_real_score) 
        fake_score_per_epoch.append(cur_fake_score) 

        loop.set_postfix(loss_g=gen_loss.item(), loss_d=loss_disc.item(), real_score=cur_real_score, fake_score=cur_fake_score)

    # Record losses & scores
    losses_g_e = np.mean(loss_g_per_epoch)
    losses_d_e = np.mean(loss_d_per_epoch)
    real_scores_e = np.mean(real_score_per_epoch)
    fake_scores_e = np.mean(fake_score_per_epoch)

    return losses_g_e, losses_d_e, real_scores_e, fake_scores_e
   

def main():
    # Initialize dataset
    dataset = MyImageFolder(root_dir=os.path.join(WORK_PATH, IMAGE_PATH))
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
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if LOAD_GEN:
        load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
    if LOAD_DISC:
        load_checkpoint(CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE)

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    for epoch in range(NUM_EPOCHS):
        losses_g_e, losses_d_e, real_scores_e, fake_scores_e = train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, epoch)
        
        losses_g.append(losses_g_e)
        losses_d.append(losses_d_e)
        real_scores.append(real_scores_e)
        fake_scores.append(fake_scores_e)
        
        if (epoch+1) % PLOT_FFREQUENCY == 0:
            print(f"Plotting samples for epoch {epoch+1}" )
            plot_examples(gen)
    
    if SAVE_MODEL:
        save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

    # Show Losses
    plt.figure(figsize=(15, 6))
    plt.plot(losses_d, '-')
    plt.plot(losses_g, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.show()

    if DISC_TRAIN:
        # Show Scores
        plt.figure(figsize=(15, 6))
        plt.plot(real_scores, '-')
        plt.plot(fake_scores, '-')
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.legend(['Real', 'Fake'])
        plt.title('Scores')
        plt.show()


if __name__ == "__main__":
    main()