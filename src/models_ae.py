import torch
import torch.nn as nn
import math

import sys
import os
from src.gino_ae import ConditionalEncoder, GINO_Encoder
import torch.nn.functional as F
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
    A HyperNetwork mapping mesh data to grid using simple IDW interpolation,
    followed by DiT blocks and CNN downsampling for compression.
    """
    def __init__(self, 
                 t_chunk=16,
                 channel_in=2, 
                 coord_dim=2, 
                 latent_dim=256, 
                 hidden_dim=256, 
                 num_heads=8, 
                 depth=4,
                 patch_size=2,
                 use_gino=False,
                 gno_radius=0.05):
        super().__init__()
        self.t_chunk = t_chunk
        self.patch_size = patch_size
        self.use_gino = use_gino
        
        freq_in_channels = (t_chunk // 2 + 1) * 2 * channel_in
        
        self.latent_grid_size = 64
        self.resolution = 16  # After downsampling

        # 1. Map input to hidden dim
        self.input_proj = nn.Linear(freq_in_channels, hidden_dim)
        
        self.register_buffer("latent_grid", self.get_latent_grid(self.latent_grid_size))
        
        if self.use_gino:
            self.gino_encoder = GINO_Encoder(
                in_channels=freq_in_channels,
                projection_channels=hidden_dim,
                gno_coord_dim=coord_dim,
                gno_radius=gno_radius,
                gno_mlp_hidden_layers=[80, 80, 80],
                gno_mlp_non_linearity=F.gelu,
                gno_transform_type='linear',
            )

        # 2. DiT processing
        self.dit = DiT(
            input_size=[self.latent_grid_size, self.latent_grid_size],
            patch_size=[patch_size, patch_size],
            in_channels=hidden_dim,
            hidden_size=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            context_dim=None,
            dim=2
        )

        # 3. CNN compression (2 downsamples to go from 64x64 to 16x16)
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        )

        self.null_context = nn.Parameter(torch.zeros(1, 1, self.dit.t_embedder.mlp[2].out_features))

    def get_latent_grid(self, N):
        xx = torch.linspace(-1, 1, N)
        yy = torch.linspace(-1, 1, N)

        xx, yy = torch.meshgrid(xx, yy, indexing='ij')
        latent_queries = torch.stack([xx, yy], dim=-1)
        
        return latent_queries.view(-1, 2).unsqueeze(0)

    def interpolate_to_grid(self, x_N, coords, grid):
        B, N, C = x_N.shape
        M = grid.shape[1]
        
        dist = torch.cdist(grid.expand(B, -1, -1), coords)
        weights, indices = torch.topk(dist, k=3, dim=-1, largest=False)
        weights = 1.0 / (weights**2 + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Memory efficient retrieval avoiding massive tensor expansion
        batch_idx = torch.arange(B, device=x_N.device).view(B, 1, 1)
        x_gathered = x_N[batch_idx, indices]
        
        x_grid = (x_gathered * weights.unsqueeze(-1)).sum(dim=2)
        H = int(M**0.5)
        x_grid = x_grid.transpose(1, 2).reshape(B, C, H, H)
        return x_grid

    def forward(self, x_noisy, coords, t, pad_mask=None):
        B, _, N, C = x_noisy.shape
        
        x_freq = torch.fft.rfft(x_noisy, n=self.t_chunk, dim=1, norm="ortho")
        x_freq_real = torch.view_as_real(x_freq)
        x_noisy_freq = x_freq_real.permute(0, 2, 1, 3, 4).reshape(B, N, -1)
        
        if self.use_gino:
            grid_latents_list = []
            for i in range(B):
                x_batch = x_noisy_freq[i].unsqueeze(0)
                coords_batch = coords[i].unsqueeze(0)
                latent_batch = self.gino_encoder(x_batch, coords_batch, self.latent_grid)
                grid_latents_list.append(latent_batch)
            grid_latents = torch.cat(grid_latents_list, dim=0) # [B, 64, 64, hidden_dim]
            grid_latents = grid_latents.permute(0, 3, 1, 2) # [B, hidden_dim, 64, 64]
        else:
            # 1. Map to hidden_dim
            x_hidden = self.input_proj(x_noisy_freq) # [B, N, hidden_dim]
            
            # 2. Interpolate to 64x64 grid
            grid_latents = self.interpolate_to_grid(x_hidden, coords, self.latent_grid) # [B, hidden_dim, 64, 64]
        
        # 3. DiT processing
        dummy_context = self.null_context.expand(B, 1, -1)
        x = self.dit(grid_latents, t, dummy_context) # [B, hidden_dim, 64, 64]
        
        # 4. CNN compression to 16x16
        z = self.downsample(x) # [B, latent_dim, 16, 16]
        
        # Permute to channels last if the rest of the code expects [B, H, W, C]
        z = z.permute(0, 2, 3, 1) # [B, 16, 16, latent_dim]
        
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


class HyperNetwork_GINO3D(nn.Module):
    """
    A HyperNetwork mapping 3D mesh data to grid using simple IDW interpolation,
    followed by DiT blocks and CNN downsampling for compression.
    """
    def __init__(self, 
                 t_chunk=16,
                 channel_in=2, 
                 coord_dim=3, 
                 latent_dim=256, 
                 hidden_dim=256, 
                 num_heads=8, 
                 depth=4,
                 patch_size=2,
                 use_gino=False,
                 gno_radius=0.05):
        super().__init__()
        self.t_chunk = t_chunk
        self.patch_size = patch_size
        self.use_gino = use_gino
        
        freq_in_channels = (t_chunk // 2 + 1) * 2 * channel_in
        
        self.latent_grid_size = 16
        self.resolution = 4  # After downsampling

        # 1. Map input to hidden dim
        self.input_proj = nn.Linear(freq_in_channels, hidden_dim)
        
        self.register_buffer("latent_grid", self.get_latent_grid(self.latent_grid_size))
        
        if self.use_gino:
            self.gino_encoder = GINO_Encoder(
                in_channels=freq_in_channels,
                projection_channels=hidden_dim,
                gno_coord_dim=coord_dim,
                gno_radius=gno_radius,
                gno_mlp_hidden_layers=[80, 80, 80],
                gno_mlp_non_linearity=F.gelu,
                gno_transform_type='linear',
            )

        # 2. DiT processing
        self.dit = DiT(
            input_size=[self.latent_grid_size, self.latent_grid_size, self.latent_grid_size],
            patch_size=[patch_size, patch_size, patch_size],
            in_channels=hidden_dim,
            hidden_size=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            context_dim=None,
            dim=3
        )

        # 3. CNN compression (2 downsamples to go from 16x16x16 to 4x4x4)
        self.downsample = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        )

        self.null_context = nn.Parameter(torch.zeros(1, 1, self.dit.t_embedder.mlp[2].out_features))

    def get_latent_grid(self, N):
        xx = torch.linspace(-1, 1, N)
        yy = torch.linspace(-1, 1, N)
        zz = torch.linspace(-1, 1, N)

        xx, yy, zz = torch.meshgrid(xx, yy, zz, indexing='ij')
        latent_queries = torch.stack([xx, yy, zz], dim=-1)
        
        return latent_queries.view(-1, 3).unsqueeze(0)

    def interpolate_to_grid(self, x_N, coords, grid):
        B, N, C = x_N.shape
        M = grid.shape[1]
        
        dist = torch.cdist(grid.expand(B, -1, -1), coords)
        weights, indices = torch.topk(dist, k=3, dim=-1, largest=False)
        weights = 1.0 / (weights**2 + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Memory efficient retrieval avoiding massive tensor expansion
        batch_idx = torch.arange(B, device=x_N.device).view(B, 1, 1)
        x_gathered = x_N[batch_idx, indices]
        
        x_grid = (x_gathered * weights.unsqueeze(-1)).sum(dim=2)
        H = int(round(M**(1/3)))
        x_grid = x_grid.transpose(1, 2).reshape(B, C, H, H, H)
        return x_grid

    def forward(self, x_noisy, coords, t, pad_mask=None):
        B, _, N, C = x_noisy.shape
        
        x_freq = torch.fft.rfft(x_noisy, n=self.t_chunk, dim=1, norm="ortho")
        x_freq_real = torch.view_as_real(x_freq)
        x_noisy_freq = x_freq_real.permute(0, 2, 1, 3, 4).reshape(B, N, -1)
        
        if self.use_gino:
            grid_latents_list = []
            for i in range(B):
                x_batch = x_noisy_freq[i].unsqueeze(0)
                coords_batch = coords[i].unsqueeze(0)
                latent_batch = self.gino_encoder(x_batch, coords_batch, self.latent_grid)
                grid_latents_list.append(latent_batch)
            grid_latents = torch.cat(grid_latents_list, dim=0) # [B, 16, 16, 16, hidden_dim]
            grid_latents = grid_latents.permute(0, 4, 1, 2, 3) # [B, hidden_dim, 16, 16, 16]
        else:
            # 1. Map to hidden_dim
            x_hidden = self.input_proj(x_noisy_freq) # [B, N, hidden_dim]
            
            # 2. Interpolate to 16x16x16 grid
            grid_latents = self.interpolate_to_grid(x_hidden, coords, self.latent_grid) # [B, hidden_dim, 16, 16, 16]
        
        # 3. DiT processing
        dummy_context = self.null_context.expand(B, 1, -1)
        x = self.dit(grid_latents, t, dummy_context) # [B, hidden_dim, 16, 16, 16]
        
        # 4. CNN compression to 4x4x4
        z = self.downsample(x) # [B, latent_dim, 4, 4, 4]
        
        # Permute to channels last if the rest of the code expects [B, H, W, D, C]
        z = z.permute(0, 2, 3, 4, 1) # [B, 4, 4, 4, latent_dim]
        
        return z


class GaborRenderer_GINO3D(nn.Module):
    """
    3D translation of GaborRenderer_GINO
    """
    def __init__(self, latent_dim=256, coord_dim=3, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4):
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
        
        # Positional embedding for latent grid (assuming 4x4x4 grid resolution to match downsampled latents)
        grid_size = 4
        xx = torch.linspace(-1, 1, grid_size)
        yy = torch.linspace(-1, 1, grid_size)
        zz = torch.linspace(-1, 1, grid_size)
        grid = torch.stack(torch.meshgrid(xx, yy, zz, indexing='ij'), dim=-1) # [4, 4, 4, 3]
        self.register_buffer("latent_grid", grid.view(-1, 3)) # [64, 3]
        
        # Proj for mapping positional grid to concatenate
        self.pos_proj = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
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
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim),
            )
            
        # Proper initialization for MFN FiLM modulators (net2)
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim)
            )

    def forward(self, z, coords):
        B, N, _ = coords.shape
        x0 = coords # x0 is [B, N, coord_dim], used for computing geometry basis in filters
        
        # If z has spatial dimensions, flatten them
        if z.dim() == 5:
            B_z, D, H, W, C = z.shape
            z = z.view(B_z, D * H * W, C)
            
        # Localized Attention Layer: Attend to z using coords
        # This gives a unique interpolated feature z for each coordinate point N
        if z.dim() == 3 and z.size(1) > 1:
            # Map position to hidden, concatenate with z, then map back to latent_dim
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
