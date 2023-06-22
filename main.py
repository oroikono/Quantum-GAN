# import sys
# sys.path.append('.')
from data.boson.boson_sampler import generate_boson_data
from models.gan import Generator, Critic
from evaluation.evaluation import create_cdf, plot_cdf
import torch
from torch.autograd import Variable
from torch.optim import RMSprop
from datetime import datetime
import os
import numpy as np


# Boson Sampling parameters
n_samples = 10000
n_photons = 8
n_modes = 16

# GAN architecture parameters
latent_dim = 16
hidden_dim = 512

# Training parameters
n_epochs = 40000
n_critic = 5
batch_size = 500
lr = 5e-4

def train_gan(real_data):
    generator = Generator(latent_dim, hidden_dim, n_modes)
    critic = Critic(hidden_dim,n_modes)
    opt_gen = RMSprop(generator.parameters(), lr=lr)
    opt_critic = RMSprop(critic.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Train critic
        for _ in range(n_critic):
            critic.zero_grad()

            # Real data
            real_batch = real_data[np.random.choice(n_samples, batch_size)]
            real_loss = -torch.mean(critic(real_batch))

            # Fake data
            noise = Variable(torch.randn(batch_size, latent_dim))
            fake_batch = generator(noise).detach()
            fake_loss = torch.mean(critic(fake_batch))

            # Critic loss
            critic_loss = real_loss + fake_loss
            critic_loss.backward()
            opt_critic.step()

        # Train generator
        generator.zero_grad()
        gen_loss = -torch.mean(critic(generator(noise)))
        gen_loss.backward()
        opt_gen.step()

        # Print losses
        if (epoch+1) % 2000 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Critic Loss: {critic_loss.item()}, Generator Loss: {gen_loss.item()}")
    # Save models
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"output/{current_date}"
    os.makedirs(folder_name, exist_ok=True)

    generator_path = os.path.join(folder_name, "generator.pt")
    critic_path = os.path.join(folder_name, "critic.pt")
    torch.save(generator.state_dict(), generator_path)
    torch.save(critic.state_dict(), critic_path)
    return generator

def evaluate_gan(generator):
    # Generate fake data for evaluation
    noise = Variable(torch.randn(n_samples, latent_dim))
    fake_data = generator(noise)

    # Convert to numpy
    fake_data_np = fake_data.detach().numpy()

    # Create CDF
    p, sorted_data = create_cdf(fake_data_np)

    # Plot CDF
    plot_cdf(p, sorted_data)

def main():
    # Generate boson sampling data
    real_data = generate_boson_data(n_samples, n_photons, n_modes)

    # Train GAN
    generator = train_gan(real_data)
    

    # Evaluate GAN
    evaluate_gan(generator)

if __name__ == "__main__":
    main()
