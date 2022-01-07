import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes, embedding_dim, latent_dim):
        super(Generator, self).__init__()

        self.conditional_labels = nn.Sequential(nn.Embedding(num_classes, embedding_dim), nn.Linear(embedding_dim, 16))
        self.latent = nn.Sequential(nn.Linear(latent_dim, 4*4*512), nn.LeakyReLU(0.2, inplace=True))

        self.gen = nn.Sequential(nn.ConvTranspose2d)