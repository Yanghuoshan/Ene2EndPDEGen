import torch
import torch.nn as nn
import math

import sys
import os
from src.gino_ae import ConditionalEncoder
from src.transformer import DiT


class FullGaborLayer(nn.Module):
    """
    Standard Gabor-like filter as used in MFN (Multiplicative Filter Networks).
    Incorporates both frequency sine transformations and spatial Gaussian envelopes.
    """
    def __init__(self, in_features, out_features, weight_scale=256.0, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-math.pi, math.pi)

    def forward(self, x): 
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])
    

class HyperNetwork_GINO(nn.Module):
    """
    A HyperNetwork using GINO ConditionalEncoder to map mesh data to grid,
    followed by DiT blocks for denoising, and outputs a flat latent vector.
    """
    def __init__(self, 
                 t_chunk=16,
                 channel_in=2, 
                 coord_dim=2, 
                 latent_dim=256, 
                 hidden_dim=256, 
                 num_heads=8, 
                 depth=4,
                 gino_config=None,
                 patch_size=2):
        super().__init__()
        self.t_chunk = t_chunk
        self.patch_size = patch_size
        
        # Transform time sequence to frequency domain
        freq_in_channels = (t_chunk // 2 + 1) * 2 * channel_in
        
        if gino_config is None:
            # Provide a default config compatible with ConditionalEncoder
            gino_config = {
                "ablate": False,
                "out_dim": hidden_dim,
                "encoder": {
                    "in_channels": freq_in_channels,
                    "out_channels": hidden_dim,
                    "gno_coord_dim": coord_dim,
                    "gno_coord_embed_dim": None,
                    "gno_radius": 0.05,
                    "gno_mlp_hidden_layers": [80, 80, 80],
                    "gno_mlp_non_linearity": torch.nn.functional.gelu,
                    "gno_transform_type": 'linear',
                    "gno_use_torch_scatter": True,
                    "hidden_channels": 64,
                    "ch_mult": (1, 2, 4),
                    "num_res_blocks": 2,
                    "attn_resolutions": (16, ),
                    "dropout": 0.0,
                    "resolution": 32,
                    "z_channels": hidden_dim,
                    "double_z": False,
                    "use_open3d": False,
                    "tanh_out": False
                }
            }
        
        self.latent_grid_size = 64  # User explicitly requested 64
        
        # Override config resolution to match the inherent latent grid size
        if "encoder" in gino_config:
            gino_config["encoder"]["resolution"] = self.latent_grid_size
            # Calculate the output feature map resolution after CNN downsampling
            ch_mult = gino_config["encoder"].get("ch_mult", (1, 2, 4))
            num_downsamples = len(ch_mult) - 1
            self.resolution = self.latent_grid_size // (2 ** num_downsamples)
        else:
            self.resolution = self.latent_grid_size // 4 # Default

        # 1. GINO Encoder maps from mesh to grid features (z_channels)
        self.gino_encoder = ConditionalEncoder(gino_config)
        self.register_buffer("latent_grid", self.get_latent_grid(self.latent_grid_size))
        
        # 2. DiT processing
        # Replace manual patch, time embeddings, DiTBlocks, unpatchify with transformer.DiT
        self.dit = DiT(
            input_size=[self.resolution, self.resolution],
            patch_size=[patch_size, patch_size],
            in_channels=hidden_dim,
            hidden_size=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            context_dim=None,
            dim=2
        )

        # 3. Final pooling and projection
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.null_context = nn.Parameter(torch.zeros(1, 1, self.dit.t_embedder.mlp[2].out_features))

    def get_latent_grid(self, N):
        xx = torch.linspace(-1, 1, N)
        yy = torch.linspace(-1, 1, N)

        xx, yy = torch.meshgrid(xx, yy, indexing='ij')
        latent_queries = torch.stack([xx, yy], dim=-1)
        
        return latent_queries.unsqueeze(0)

    def forward(self, x_noisy, coords, t, pad_mask=None):
        B, _, N, C = x_noisy.shape
        
        # 1. FFT on temporal dynamics: [B, T_chunk, N, C] -> [B, N, F * C * 2]
        x_freq = torch.fft.rfft(x_noisy, n=self.t_chunk, dim=1, norm="ortho") # [B, F, N, C]
        x_freq_real = torch.view_as_real(x_freq) # [B, F, N, C, 2]
        x_noisy_freq = x_freq_real.permute(0, 2, 1, 3, 4).reshape(B, N, -1) # [B, N, freq_in_channels]
        
        # Encode with GINO: mesh -> grid latents
        # coords: [B, N, D]
        grid_latents = self.gino_encoder(x_noisy_freq, input_geom=coords, latent_queries=self.latent_grid, pad_mask=pad_mask)
        # grid_latents is [B, (H*W), C] -> reshape to [B, C, H, W]
        grid_latents = grid_latents.transpose(1, 2).reshape(B, -1, self.resolution, self.resolution)
        
        # DiT processing
        # DiT expects context. We pass a learnable null context of appropriate size
        # because the internal y_embedder is nn.Identity() when context_dim is None, 
        # and it will concat conditionally to make c have length hidden_dim.
        dummy_context = self.null_context.expand(B, 1, -1)
        
        x = self.dit(grid_latents, t, dummy_context) # [B, hidden_dim, H, W]
        
        # Return to channels-last for final projection
        x = x.permute(0, 2, 3, 1) # [B, H, W, hidden_dim]
        
        # Final projection to target latent dim (e.g. 256, 512, 1024)

        z = self.final_layer(x)
        return z


class GaborRenderer_GINO(nn.Module):
    """
    Direct faithful translation of GNAutodecoder_film from the official ConditionalNeuralField repo.
    Uses an MFN (Multiplicative Filter Network) structure with Gabor layers processing pure geometry 
    and Linear layers processing the features, modulated by the Latent target Z1 via additive shifting.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Localized Attention Layer (LAL) for coords and z
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        
        # Positional embedding for latent grid (assuming 16x16 grid resolution)
        grid_size = 16
        xx = torch.linspace(-1, 1, grid_size)
        yy = torch.linspace(-1, 1, grid_size)
        grid = torch.stack(torch.meshgrid(xx, yy, indexing='ij'), dim=-1) # [16, 16, 2]
        self.register_buffer("latent_grid", grid.view(-1, 2)) # [256, 2]
        
        # Proj for mapping positional grid to concatenate
        # We project pos to a feature size, then concat with z, and project back to latent_dim
        self.pos_proj = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU()
        )
        self.z_fuse_proj = nn.Linear(latent_dim + hidden_dim // 2, latent_dim)
        
        # MFN basis filters processing pure spatial geometry
        self.filters = nn.ModuleList([
            FullGaborLayer(
                in_features=coord_dim, 
                out_features=hidden_dim, 
                weight_scale=256.0 / math.sqrt(num_layers + 1), 
                alpha=6.0 / (num_layers + 1), 
                beta=1.0
            ) 
            for _ in range(num_layers + 1)
        ])
        
        # Backbone processing modulated spatial features
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        
        # Modulator projecting Latents to per-layer additive shifts
        self.net2 = nn.ModuleList(
            [nn.Linear(latent_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)]
        )

        # Output layer predicts frequency domain in real format (real+imag per component)
        self.freq_dim = (t_chunk // 2 + 1)
        self.final_linear = nn.Linear(hidden_dim, self.freq_dim * 2 * channel_out)
        
        # Proper initialization for MFN Linear Backbone
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            # weight_scale = 1.0 (default for net2 in original repo)
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim),
            )
            
        # Proper initialization for MFN FiLM modulators (net2)
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            # weight_scale = 1.0 (default for net2 in original repo)
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim)
            )

    def forward(self, z, coords):
        B, N, _ = coords.shape
        x0 = coords # x0 is [B, N, coord_dim], used for computing geometry basis in filters
        
        # If z has spatial dimensions, flatten them
        if z.dim() == 4:
            B_z, H, W, C = z.shape
            z = z.view(B_z, H * W, C)
            
        # Localized Attention Layer: Attend to z using coords
        # This gives a unique interpolated feature z for each coordinate point N
        if z.dim() == 3 and z.size(1) > 1:
            # Map position to hidden, concatenate with z, then map back to latent_dim
            # Assumes z.size(1) == self.latent_grid.size(0) (e.g. 256)
            if z.size(1) == self.latent_grid.size(0):
                pos_emb = self.pos_proj(self.latent_grid).unsqueeze(0).expand(B, -1, -1)
                z = torch.cat([z, pos_emb], dim=-1)
                z = self.z_fuse_proj(z)
                
            z_kv = self.norm_kv(z)
            q = self.norm_q(self.coord_proj(coords))
            z, _ = self.cross_attn(query=q, key=z_kv, value=z_kv) # z becomes [B, N, latent_dim]
        elif z.dim() == 2:
            z = z.unsqueeze(1) # [B, 1, latent_dim]
            
        # Step 0: Initial layer injection
        x = self.filters[0](x0) * self.net2[0](z) # [B, N, hidden_dim]
        
        # Multiplicative Filter Network Steps
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0) # Compute coordinate geometry basis
            
            # Backbone processing (Wx+b) + Latent Modulation (shift)
            h = self.net1[i - 1](x) + self.net2[i](z)
            
            # MFN Multiplication
            x = basis * h
            
        # Final output projection
        out = self.final_linear(x) # [B, N, F * 2 * channel_out]
        
        # Reshape to expected sequence shape: [B, F, N, channel_out, 2]
        out_freq_real = out.view(B, N, self.freq_dim, self.channel_out, 2)
        out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4)
        out_freq_complex = torch.view_as_complex(out_freq_real.to(torch.float32)) # [B, F, N, channel_out]

        # Convert frequency domain back to time domain
        out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        return out
