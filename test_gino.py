import torch
import sys
import os
from src.models_ae import HyperNetwork_GINO, GaborRenderer_GINO

def test():
    device = "cpu"
    x_noisy = torch.randn(2, 16, 300, 2).to(device)
    coords = torch.rand(2, 300, 2).to(device) * 2 - 1
    t = torch.tensor([0.1, 0.5]).to(device)
    
    encoder = HyperNetwork_GINO(t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, hidden_dim=256).to(device)
    renderer = GaborRenderer_GINO(latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256).to(device)
    
    z = encoder(x_noisy, coords, t)
    print("Z shape:", z.shape)
    
    out = renderer(z, coords)
    print("Out shape:", out.shape)
    
test()
