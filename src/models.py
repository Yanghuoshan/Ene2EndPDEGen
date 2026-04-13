import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    """Adaptive Layer Norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: [B]
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Init standard zero for adaLN outputs (forces block to act as identity initially)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention with Modulation
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with Modulation
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x2)
        return x


class CrossDiTBlock(nn.Module):
    """
    A cross-attention DiT block with adaLN-Zero conditioning.
    Query comes from latent tokens x, key/value comes from node features kv.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Zero init keeps residual branch near identity at startup.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, kv, c):
        shift_q, scale_q, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Cross-attention with adaLN-modulated latent queries.
        q = modulate(self.norm_q(x), shift_q, scale_q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, kv_norm, kv_norm)
        x = x + gate_attn.unsqueeze(1) * attn_out

        # MLP branch with adaLN modulation.
        mlp_in = modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x

class HyperNetwork(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps local noisy trajectories + coords to a hidden representation
        self.node_proj = nn.Sequential(
            nn.Linear(t_chunk * channel_in + coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # M Learnable tokens query global information
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        
        # Cross Attention: Global Tokens <- attend to -> Spatial Notes
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        
        # Powerful Sequence modeling via DiT backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # Flat tokens to Z1
        self.final_proj = nn.Sequential(
            nn.LayerNorm(num_tokens * hidden_dim),
            nn.Linear(num_tokens * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x_noisy, coords, t):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Extract node features ---> only spatial info, no temporal dimension! 
        # [B, N, T*C]
        x_reshaped = x_noisy.permute(0, 2, 1, 3).reshape(B, N, T * C)
        node_features = torch.cat([x_reshaped, coords], dim=-1)
        feat = self.node_proj(node_features) # [B, N, hidden_dim]
        
        # 3. Readout via Cross-Attention
        q = self.query_tokens.expand(B, -1, -1)  # [B, M, hidden_dim]
        latents, _ = self.cross_attn(self.norm_q(q), self.norm_k(feat), self.norm_k(feat))
        latents = q + latents # Res connection
        
        # 4. Processing latents through DiT conditionally
        for block in self.blocks:
            latents = block(latents, c)
            
        # 5. Output robust global Latent Space Z1
        latents_flat = latents.reshape(B, -1) # [B, M * hidden_dim]
        z1 = self.final_proj(latents_flat)    # [B, latent_dim]
        
        return z1



class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_features, mapping_size) * scale)

    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B # [batch_size, num_points, mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [batch_size, num_points, mapping_size * 2]

class HyperNetwork_FA(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        
        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps local noisy trajectories + coords to a hidden representation
        self.node_proj = nn.Sequential(
            nn.Linear(t_chunk * channel_in + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # M Learnable tokens query global information
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))
        
        # Cross Attention: Global Tokens <- attend to -> Spatial Notes
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)

        # 在 __init__ 中增加一个专用 MLP
        self.cross_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Powerful Sequence modeling via DiT backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # Flat tokens to Z1
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x_noisy, coords, t):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Extract node features ---> only spatial info, no temporal dimension! 
        # [B, N, T*C]
        x_reshaped = x_noisy.permute(0, 2, 1, 3).reshape(B, N, T * C)
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
        node_features = torch.cat([x_reshaped, coords_encoded], dim=-1)
        feat = self.node_proj(node_features) # [B, N, hidden_dim]
        
        # 3. Readout via Cross-Attention
        q = self.query_tokens.expand(B, -1, -1)  # [B, M, hidden_dim]
        q = q + c.unsqueeze(1) # add time conditioning to queries before cross-attention
        latents, _ = self.cross_attn(self.norm_q(q), self.norm_k(feat), self.norm_k(feat))
        latents = q + latents # Res connection
        latents = latents + self.cross_mlp(latents) # MLP after cross-attention for better feature extraction
        
        # 4. Processing latents through DiT conditionally
        for block in self.blocks:
            latents = block(latents, c)
            
        # 5. Output robust global Latent Space Z1
        z1 = latents.mean(dim=1) # [B, M, hidden_dim] -> [B, hidden_dim] via mean pooling
        z1 = self.final_proj(z1)    # [B, latent_dim]
        
        return z1


class HyperNetwork_AP(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        
        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps local noisy trajectories + coords to a hidden representation
        self.node_proj = nn.Sequential(
            nn.Linear(t_chunk * channel_in + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # M Learnable tokens query global information
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))
        
        # Cross Attention: Global Tokens <- attend to -> Spatial Notes
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)

        # 在 __init__ 中增加一个专用 MLP
        self.cross_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Powerful Sequence modeling via DiT backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])

        # Attention Pooling projection to get a single global representation from M tokens
        self.pool_proj = nn.Linear(hidden_dim, 1)
        
        # Flat tokens to Z1
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x_noisy, coords, t):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Extract node features ---> only spatial info, no temporal dimension! 
        # [B, N, T*C]
        x_reshaped = x_noisy.permute(0, 2, 1, 3).reshape(B, N, T * C)
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
        node_features = torch.cat([x_reshaped, coords_encoded], dim=-1)
        feat = self.node_proj(node_features) # [B, N, hidden_dim]
        
        # 3. Readout via Cross-Attention
        q = self.query_tokens.expand(B, -1, -1)  # [B, M, hidden_dim]
        q = q + c.unsqueeze(1) # add time conditioning to queries before cross-attention
        latents, _ = self.cross_attn(self.norm_q(q), self.norm_k(feat), self.norm_k(feat))
        latents = q + latents # Res connection
        latents = latents + self.cross_mlp(latents) # MLP after cross-attention for better feature extraction
        
        # 4. Processing latents through DiT conditionally
        for block in self.blocks:
            latents = block(latents, c)
            
        # 5. Attention Pooling to get a single global representation from M tokens
        attn_weights = torch.softmax(self.pool_proj(latents), dim=1)  # [B, M, 1]
        z1 = (latents * attn_weights).sum(dim=1)                      # [B, hidden_dim]

        # 6. Output robust global Latent Space Z1
        z1 = self.final_proj(z1)    # [B, latent_dim]
        
        return z1


class HyperNetwork_ST(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        
        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps local noisy trajectories + coords to a hidden representation
        self.node_proj = nn.Sequential(
            nn.Linear(t_chunk * channel_in + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # M Learnable tokens query global information
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))
        
        # Cross Attention: Global Tokens <- attend to -> Spatial Notes
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)

        # 在 __init__ 中增加一个专用 MLP
        self.cross_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Powerful Sequence modeling via DiT backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])

        # Attention Pooling projection to get a single global representation from M tokens
        self.pool_proj = nn.Linear(hidden_dim, 1)
        
        # Flat tokens to Z1
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim) # Final LayerNorm for better stability in downstream conditioning
        )

    def forward(self, x_noisy, coords, t):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Extract node features ---> only spatial info, no temporal dimension! 
        # [B, N, T*C]
        x_reshaped = x_noisy.permute(0, 2, 1, 3).reshape(B, N, T * C)
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
        node_features = torch.cat([x_reshaped, coords_encoded], dim=-1)
        feat = self.node_proj(node_features) # [B, N, hidden_dim]
        
        # 3. Readout via Cross-Attention
        q = self.query_tokens.expand(B, -1, -1)  # [B, M, hidden_dim]
        q = q + c.unsqueeze(1) # add time conditioning to queries before cross-attention
        latents, _ = self.cross_attn(self.norm_q(q), self.norm_k(feat), self.norm_k(feat))
        latents = q + latents # Res connection
        latents = latents + self.cross_mlp(latents) # MLP after cross-attention for better feature extraction
        
        # 4. Processing latents through DiT conditionally
        for block in self.blocks:
            latents = block(latents, c)
            
        # 5. Attention Pooling to get a single global representation from M tokens
        attn_weights = torch.softmax(self.pool_proj(latents), dim=1)  # [B, M, 1]
        z1 = (latents * attn_weights).sum(dim=1)                      # [B, hidden_dim]

        # 6. Output robust global Latent Space Z1
        z1 = self.final_proj(z1)    # [B, latent_dim]
        
        return z1


class HyperNetwork_Perceiver(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim

        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps local noisy trajectories + encoded coords to node features.
        self.node_proj = nn.Sequential(
            nn.Linear(t_chunk * channel_in + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable latent array for Perceiver-style processing.
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))

        # Pure Perceiver stack: alternating Cross-Attn and Self-Attn blocks.
        self.cross_blocks = nn.ModuleList([
            CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.self_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])

        # Attention Pooling projection to get a single global representation from M tokens
        self.pool_proj = nn.Linear(hidden_dim, 1)
        
        # Flat tokens to Z1
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim) # Final LayerNorm for better stability in downstream conditioning
        )

    def forward(self, x_noisy, coords, t):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Build node features from trajectory and Fourier-encoded coordinates.
        x_reshaped = x_noisy.permute(0, 2, 1, 3).reshape(B, N, T * C)
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
        node_features = torch.cat([x_reshaped, coords_encoded], dim=-1)
        feat = self.node_proj(node_features) # [B, N, hidden_dim]
        
        # 3. Perceiver latent initialization.
        q = self.query_tokens.expand(B, -1, -1)  # [B, M, hidden_dim]
        latents = q
        
        # 4. Alternate Cross-Attn and Self-Attn; each cross layer re-reads node features.
        for cross_block, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross_block(latents, feat, c)
            latents = self_block(latents, c)
            
        # 5. Attention Pooling to get a single global representation from M tokens
        attn_weights = torch.softmax(self.pool_proj(latents), dim=1)  # [B, M, 1]
        z1 = (latents * attn_weights).sum(dim=1)                      # [B, hidden_dim]

        # 6. Output robust global Latent Space Z1
        z1 = self.final_proj(z1)    # [B, latent_dim]
        
        return z1


class GaborLayer(nn.Module):
    """
    Gabor layer used in highly robust INRs: exp(-alpha * (Wx+b)**2) * sin(Wx+b)
    """
    def __init__(self, in_features, out_features, weight_scale=256.0, alpha=6.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = alpha
        
        # Special initialization suitable for Gabor waves
        nn.init.uniform_(self.linear.weight, -weight_scale / in_features, weight_scale / in_features)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -torch.pi, torch.pi)
            
    def forward(self, x):
        wx = self.linear(x)
        return torch.exp(-(wx ** 2) / (self.alpha ** 2)) * torch.sin(wx)


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


class GaborRenderer(nn.Module):
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

        # Output layer
        self.final_linear = nn.Linear(hidden_dim, t_chunk * channel_out)
        
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

    def forward(self, z1, coords):
        B, N, _ = coords.shape
        x0 = coords
        
        # Step 0: Initial layer injection
        x = self.filters[0](x0) * self.net2[0](z1).unsqueeze(1) # [B, N, hidden_dim]
        
        # Multiplicative Filter Network Steps
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0) # Compute coordinate geometry basis
            
            # Backbone processing (Wx+b) + Latent Modulation (shift)
            h = self.net1[i - 1](x) + self.net2[i](z1).unsqueeze(1)
            
            # MFN Multiplication
            x = basis * h
            
        # Final output projection
        out = self.final_linear(x) # [B, N, t_chunk * channel_out]
        
        # Reshape to expected sequence shape: [B, T_chunk, N, channel_out]
        out = out.view(B, N, self.t_chunk, self.channel_out)
        out = out.permute(0, 2, 1, 3) 
        return out


class CNFRenderer(nn.Module):
    """
    GNAutodecoder_film equivalent architecture.
    A Gabor Network whose intermediate layers are heavily modulated (FiLM) by the Latent target Z1.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Gabor Layers map solely the space geometry
        self.layers = nn.ModuleList([
            GaborLayer(coord_dim if i == 0 else hidden_dim, hidden_dim, weight_scale=256.0 if i==0 else 1.0) 
            for i in range(num_layers)
        ])
        
        # Latent Z_1 -> Parameter Controller (Predicts Multiplicative Shift / Scale for all Gabor Layers)
        # Returns 2 values per layer (gamma and beta)
        self.film_gen = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_layers * hidden_dim * 2)
        )
        
        # Outputs all time steps natively -> True 4D dynamics synthesis
        self.final_linear = nn.Linear(hidden_dim, t_chunk * channel_out)
        
    def forward(self, z1, coords):
        B, N, _ = coords.shape
        
        # 1. Generate Conditional Gamma and Beta for FiLM
        film_params = self.film_gen(z1) # [B, num_layers * 2 * hidden_dim]
        # Reshape to [B, num_layers, 2, hidden_dim]
        film_params = film_params.view(B, self.num_layers, 2, self.hidden_dim)
        
        # Geometry coordinates input
        x = coords
        
        # 2. Iterate Geometry through Field, condition firmly with Latents
        for i, layer in enumerate(self.layers):
            # Base Coordinate Gabor Encoding
            x = layer(x)
            
            # Apply Temporal/Physics dynamics (Z1) conditionally onto space! (FiLM)
            gamma = film_params[:, i, 0, :].unsqueeze(1) # [B, 1, hidden_dim]
            beta  = film_params[:, i, 1, :].unsqueeze(1) # [B, 1, hidden_dim]
            
            # Affine Modulation
            x = x * (1.0 + gamma) + beta
        
        # 3. Get entire Sequence Output
        out = self.final_linear(x) # [B, N, t_chunk * channel_out]
        
        # Transform back to Expected Video Shape: [B, T_chunk, N, channel_out]
        out = out.view(B, N, self.t_chunk, self.channel_out)
        out = out.permute(0, 2, 1, 3) 
        
        return out

class HyperNetwork_MultiLatentPerceiver(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points.
    Returns latents of shape [B, M, latent_dim] rather than pooling them into a single vector.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim

        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps local noisy trajectories + encoded coords to node features.
        self.node_proj = nn.Sequential(
            nn.Linear(t_chunk * channel_in + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable latent array for Perceiver-style processing.
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))

        # Pure Perceiver stack: alternating Cross-Attn and Self-Attn blocks.
        self.cross_blocks = nn.ModuleList([
            CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.self_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # Flat tokens to Z1
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim) # Final LayerNorm for better stability in downstream conditioning
        )

    def forward(self, x_noisy, coords, t):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Build node features from trajectory and Fourier-encoded coordinates.
        x_reshaped = x_noisy.permute(0, 2, 1, 3).reshape(B, N, T * C)
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
        node_features = torch.cat([x_reshaped, coords_encoded], dim=-1)
        feat = self.node_proj(node_features) # [B, N, hidden_dim]
        
        # 3. Perceiver latent initialization.
        q = self.query_tokens.expand(B, -1, -1)  # [B, M, hidden_dim]
        latents = q
        
        # 4. Alternate Cross-Attn and Self-Attn; each cross layer re-reads node features.
        for cross_block, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross_block(latents, feat, c)
            latents = self_block(latents, c)
            
        # 5. Output robust global Latent Space Z1
        # No attention pooling, output is [B, M, latent_dim]
        z1 = self.final_proj(latents)    # [B, M, latent_dim]
        
        return z1


class SpatialTemporalRenderer(nn.Module):
    """
    Spatio-Temporal MFN Gabor Renderer. Takes continuous spatial and temporal coordinates.
    Latent Z1 is expected to be [B, M, latent_dim].
    """
    def __init__(self, num_tokens=16, latent_dim=256, coord_dim=2, time_dim=1, channel_out=2, hidden_dim=256, num_layers=4):
        super().__init__()
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_features = coord_dim + time_dim  # Space + Time
        
        # Latent representation is flattened M tokens
        z_dim = num_tokens * latent_dim

        # MFN basis filters processing pure spatial geometry
        self.filters = nn.ModuleList([
            FullGaborLayer(
                in_features=self.in_features, 
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
            [nn.Linear(z_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)]
        )

        # Output layer
        self.final_linear = nn.Linear(hidden_dim, channel_out)
        
        # Proper initialization for MFN Linear Backbone
        for lin in self.net1:
            in_dim_w = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim_w),
                math.sqrt(1.0 / in_dim_w),
            )
            
        # Proper initialization for MFN FiLM modulators (net2)
        for lin in self.net2:
            in_dim_w = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim_w),
                math.sqrt(1.0 / in_dim_w)
            )

    def forward(self, z1, coords, time_steps):
        # z1: [B, M, latent_dim]
        # coords: [B, N, coord_dim]
        # time_steps: [T]
        B, N, coord_dim = coords.shape
        T = time_steps.shape[0]
        
        # Create full space-time coordinates grid [B, T, N, coord_dim + 1]
        coords_exp = coords.unsqueeze(1).expand(B, T, N, coord_dim)
        time_exp = time_steps.view(1, T, 1, 1).expand(B, T, N, 1)
        
        x0 = torch.cat([coords_exp, time_exp], dim=-1) # [B, T, N, in_features]
        x0_flat = x0.reshape(B, T * N, self.in_features) # [B, T * N, in_features]
        
        # Flatten z1 for modulation
        z1_flat = z1.reshape(B, -1) # [B, M * latent_dim]
        
        # Step 0: Initial layer injection
        x = self.filters[0](x0_flat) * self.net2[0](z1_flat).unsqueeze(1) # [B, T * N, hidden_dim]
        
        # Multiplicative Filter Network Steps
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0_flat) # Compute coordinate geometry basis
            
            # Backbone processing (Wx+b) + Latent Modulation (shift)
            h = self.net1[i - 1](x) + self.net2[i](z1_flat).unsqueeze(1)
            
            # MFN Multiplication
            x = basis * h
            
        # Final output projection
        out = self.final_linear(x) # [B, T * N, channel_out]
        
        # Reshape to expected sequence shape: [B, T, N, channel_out]
        out = out.reshape(B, T, N, self.channel_out)
        return out
