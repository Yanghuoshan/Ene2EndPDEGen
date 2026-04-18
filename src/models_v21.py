import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.siren import SIRENRenderer


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


class FlashDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning using Flash Attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
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
        B, N, C = norm_x.shape
        qkv = self.qkv(norm_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with Modulation
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x2)
        return x


class FlashCrossDiTBlock(nn.Module):
    """
    A cross-attention DiT block with adaLN-Zero conditioning using Flash Attention.
    Query comes from latent tokens x, key/value comes from node features kv.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.proj = nn.Linear(hidden_size, hidden_size)
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

    def forward(self, x, kv_input, c):
        shift_q, scale_q, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Cross-attention with adaLN-modulated latent queries.
        q_norm = modulate(self.norm_q(x), shift_q, scale_q)
        kv_norm = self.norm_kv(kv_input)
        
        B, M, C = q_norm.shape
        q = self.q_proj(q_norm).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        B, N_kv, C = kv_norm.shape
        kv = self.kv_proj(kv_norm).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, M, C)
        attn_out = self.proj(attn_out)
        
        x = x + gate_attn.unsqueeze(1) * attn_out

        # MLP branch with adaLN modulation.
        mlp_in = modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class HyperNetwork_Perceiver_Flash(nn.Module):
    """
    DiT-style Set Encoder using Flash Attention blocks completely.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, encoded_coord_dim=128, use_node_type=False, node_type_dim=16):
        super().__init__()
        self.t_chunk = t_chunk
        self.use_node_type = use_node_type
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        freq_dim = (t_chunk // 2 + 1) * 2 * channel_in
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        in_dim_node = hidden_dim + encoded_coord_dim + (node_type_dim if use_node_type else 0)
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim_node, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))

        self.cross_blocks = nn.ModuleList([
            FlashCrossDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.self_blocks = nn.ModuleList([
            FlashDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x_noisy, coords_encoded, t, type_embeds=None):
        B, T, N, C = x_noisy.shape
        
        c = self.time_mlp(t)
        
        x_freq = torch.fft.rfft(x_noisy.float(), dim=1, norm="ortho")
        x_freq_real = torch.view_as_real(x_freq).permute(0, 2, 1, 3, 4).reshape(B, N, -1).type_as(x_noisy)
        freq_features = self.freq_proj(x_freq_real)
        
        if self.use_node_type and type_embeds is not None:
            node_features = torch.cat([freq_features, coords_encoded, type_embeds], dim=-1)
        else:
            node_features = torch.cat([freq_features, coords_encoded], dim=-1)
            
        feat = self.node_proj(node_features)
        
        q = self.query_tokens.expand(B, -1, -1)
        latents = q
        
        for cross_block, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross_block(latents, feat, c)
            latents = self_block(latents, c)
            
        z_multi = self.final_proj(latents)
        
        return z_multi


class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_features, mapping_size) * scale)

    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B # [batch_size, num_points, mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # [batch_size, num_points, mapping_size * 2]


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


class HyperNetwork_Perceiver_v2(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    When t_chunk is 128, the prediction is worse than t_chunk=16, likely due to the increased difficulty of learning stable Fourier features and attention over longer sequences.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, encoded_coord_dim=128, use_node_type=False, node_type_dim=16):
        super().__init__()
        self.t_chunk = t_chunk
        self.use_node_type = use_node_type
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Convert time domain [B, T, N, C] to frequency domain
        freq_dim = (t_chunk // 2 + 1) * 2 * channel_in # t_chunk = 64 -> freq_dim = 33 * 2 * 2 = 132
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps temporal features + encoded coords to node features.
        in_dim_node = hidden_dim + encoded_coord_dim + (node_type_dim if use_node_type else 0)
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim_node, hidden_dim),
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
        
        # Flat tokens to Z
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim) # Final LayerNorm for better stability in downstream conditioning
        )

    def forward(self, x_noisy, coords_encoded, t, type_embeds=None):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Build node features from trajectory and Fourier-encoded coordinates.
        # rfft requires float32, then cast back real part to bfloat16
        x_freq = torch.fft.rfft(x_noisy.float(), dim=1, norm="ortho")  # [B, F, N, C]
        x_freq_real = torch.view_as_real(x_freq).permute(0, 2, 1, 3, 4).reshape(B, N, -1).type_as(x_noisy)  # [B, N, F * C * 2]
        freq_features = self.freq_proj(x_freq_real) # [B, N, hidden_dim]
        
        if self.use_node_type and type_embeds is not None:
            node_features = torch.cat([freq_features, coords_encoded, type_embeds], dim=-1)
        else:
            node_features = torch.cat([freq_features, coords_encoded], dim=-1)
            
        feat = self.node_proj(node_features) # [B, N, hidden_dim]
        
        # 3. Perceiver latent initialization.
        q = self.query_tokens.expand(B, -1, -1)  # [B, M, hidden_dim]
        latents = q
        
        # 4. Alternate Cross-Attn and Self-Attn; each cross layer re-reads node features.
        for cross_block, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross_block(latents, feat, c)
            latents = self_block(latents, c)
            
        # 5. Output multiple robust latents from M tokens
        z_multi = self.final_proj(latents)    # [B, M, latent_dim]
        
        return z_multi


class GaborRenderer_v2(nn.Module):
    """
    Direct faithful translation of GNAutodecoder_film from the official ConditionalNeuralField repo.
    Uses an MFN (Multiplicative Filter Network) structure with Gabor layers processing pure geometry 
    and Linear layers processing the features, modulated by the Latent target Z1 via additive shifting.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, use_node_type=False, node_type_dim=16, encoded_coord_dim=128):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.use_node_type = use_node_type
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

        # Cross-attention to extract coordinate-specific latents
        in_dim_query = encoded_coord_dim + (node_type_dim if use_node_type else 0)
        self.query_proj = nn.Sequential(
            nn.Linear(in_dim_query, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

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

    def forward(self, z_multi, coords, coords_encoded, type_embeds=None):
        B, N, _ = coords.shape
        x0 = coords # x0 is [B, N, coord_dim], used for computing geometry basis in filters and as queries in cross-attention
        
        # Extract location-specific latent via Cross-Attention
        if self.use_node_type and type_embeds is not None:
            query_input = torch.cat([coords_encoded, type_embeds], dim=-1)
        else:
            query_input = coords_encoded
            
        q = self.query_proj(query_input)
        q = self.norm_q(q)
        kv = self.norm_kv(z_multi)
        z, _ = self.cross_attn(q, kv, kv) # z is [B, N, latent_dim]
        
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
        out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4).float()
        out_freq_complex = torch.view_as_complex(out_freq_real) # [B, F, N, channel_out]

        # Convert frequency domain back to time domain
        out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        return out

class FullModel_v21(nn.Module):
    """
    End-to-End Model encapsulating the shared Fourier and Node Type Embeddings,
    as well as the HyperNetwork Encoder and GaborRenderer/SIRENRenderer Decoder.
    """
    def __init__(self, t_chunk=16, channel_in=2, channel_out=2, coord_dim=2, latent_dim=256, 
                 time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, 
                 fourier_dim=64, num_layers=4, use_node_type=False, num_node_types=6, node_type_dim=16,
                 renderer_type='gabor'):
        super().__init__()
        self.use_node_type = use_node_type
        
        # 1. Shared Feature Extractors
        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        if self.use_node_type:
            self.node_type_embed = nn.Embedding(num_node_types, node_type_dim)
            
        # 2. Sub-modules
        self.encoder = HyperNetwork_Perceiver_v2(
            t_chunk=t_chunk, channel_in=channel_in, coord_dim=coord_dim, latent_dim=latent_dim,
            time_emb_dim=time_emb_dim, hidden_dim=hidden_dim, num_heads=num_heads, depth=depth, 
            num_tokens=num_tokens, encoded_coord_dim=fourier_dim*2, use_node_type=use_node_type, 
            node_type_dim=node_type_dim
        )
        
        if renderer_type == 'siren':
            self.decoder = SIRENRenderer(
                latent_dim=latent_dim, coord_dim=coord_dim, t_chunk=t_chunk, channel_out=channel_out, 
                hidden_dim=hidden_dim, num_layers=num_layers, use_node_type=use_node_type, 
                node_type_dim=node_type_dim, encoded_coord_dim=fourier_dim*2
            )
            print("Using SIREN Renderer")
        else:
            self.decoder = GaborRenderer_v2(
                latent_dim=latent_dim, coord_dim=coord_dim, t_chunk=t_chunk, channel_out=channel_out, 
                hidden_dim=hidden_dim, num_layers=num_layers, use_node_type=use_node_type, 
                node_type_dim=node_type_dim, encoded_coord_dim=fourier_dim*2
            )
            print("Using Gabor Renderer")

    def forward(self, x_noisy, t, input_coords, query_coords, input_node_type=None, query_node_type=None):
        # 1. Encode coordinates
        input_coords_encoded = self.coord_encoder(input_coords)
        query_coords_encoded = self.coord_encoder(query_coords)
        
        # 2. Embed node types
        input_type_embeds = None
        query_type_embeds = None
        if self.use_node_type:
            if input_node_type is not None:
                input_type_embeds = self.node_type_embed(input_node_type.squeeze(-1).long())
            if query_node_type is not None:
                query_type_embeds = self.node_type_embed(query_node_type.squeeze(-1).long())
                
        # 3. Encoder extracts latent
        z_multi = self.encoder(x_noisy, input_coords_encoded, t, input_type_embeds)
        
        # 4. Decoder renders predictions
        out = self.decoder(z_multi, query_coords, query_coords_encoded, query_type_embeds)
        
        return out

class AttentionDecoderBlock(nn.Module):
    """
    A standard attention block with Cross-Attention followed by Self-Attention and an MLP.
    Query comes from the encoded coordinates, key/value comes from the latent tokens in Cross-Attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj_cross = nn.Linear(hidden_size, hidden_size)
        self.kv_proj_cross = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_cross = nn.Linear(hidden_size, hidden_size)
        
        self.norm_self = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, q, kv):
        # Cross-attention: query is x (coords), kv is from latent z_multi
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        
        B, N, C = q_norm.shape
        q_c = self.q_proj_cross(q_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        B, M, C = kv_norm.shape
        kv_c = self.kv_proj_cross(kv_norm).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_c, v_c = kv_c.unbind(0)
        
        attn_out = F.scaled_dot_product_attention(q_c, k_c, v_c)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj_cross(attn_out)
        
        q = q + attn_out

        # Self-attention using Flash Attention (SDPA)
        q_norm2 = self.norm_self(q)
        B, N, C = q_norm2.shape
        qkv = self.qkv(q_norm2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_sa, k_sa, v_sa = qkv.unbind(0)
        
        attn_out2 = F.scaled_dot_product_attention(q_sa, k_sa, v_sa)
        attn_out2 = attn_out2.transpose(1, 2).reshape(B, N, C)
        attn_out2 = self.proj(attn_out2)
        
        q = q + attn_out2

        # MLP branch
        mlp_in = self.norm_mlp(q)
        q = q + self.mlp(mlp_in)
        return q


class AttentionRenderer(nn.Module):
    """
    Decoder module using stacked Cross-Attention and Self-Attention blocks.
    The query points and their types (encoded) act as the initial sequence, 
    cross-attending directly to the latent tokens (z_multi) from the encoder.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, use_node_type=False, node_type_dim=16, encoded_coord_dim=128, num_heads=8):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.use_node_type = use_node_type
        
        in_dim_query = encoded_coord_dim + (node_type_dim if use_node_type else 0)
        self.query_proj = nn.Sequential(
            nn.Linear(in_dim_query, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if latent_dim != hidden_dim else nn.Identity()

        self.blocks = nn.ModuleList([
            AttentionDecoderBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.freq_dim = (t_chunk // 2 + 1)
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, self.freq_dim * 2 * channel_out)

    def forward(self, z_multi, coords, coords_encoded, type_embeds=None):
        B, N, _ = coords.shape
        
        if self.use_node_type and type_embeds is not None:
            query_input = torch.cat([coords_encoded, type_embeds], dim=-1)
        else:
            query_input = coords_encoded
            
        q = self.query_proj(query_input)  # [B, N, hidden_dim]
        kv = self.latent_proj(z_multi)    # [B, M, hidden_dim]
        
        for block in self.blocks:
            q = block(q, kv)
            
        q = self.norm_final(q)
        out = self.final_linear(q) # [B, N, F * 2 * channel_out]
        
        out_freq_real = out.view(B, N, self.freq_dim, self.channel_out, 2)
        out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4).float()
        out_freq_complex = torch.view_as_complex(out_freq_real) # [B, F, N, channel_out]

        # Convert frequency domain back to time domain
        out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        return out


class FullModel_Attention(nn.Module):
    """
    End-to-End Model containing the HyperNetwork Encoder and an AttentionRenderer Decoder.
    """
    def __init__(self, t_chunk=16, channel_in=2, channel_out=2, coord_dim=2, latent_dim=256, 
                 time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, 
                 fourier_dim=64, num_layers=4, use_node_type=False, num_node_types=6, node_type_dim=16):
        super().__init__()
        self.use_node_type = use_node_type
        
        # 1. Shared Feature Extractors
        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        if self.use_node_type:
            self.node_type_embed = nn.Embedding(num_node_types, node_type_dim)
            
        # 2. Sub-modules
        self.encoder = HyperNetwork_Perceiver_Flash(
            t_chunk=t_chunk, channel_in=channel_in, coord_dim=coord_dim, latent_dim=latent_dim,
            time_emb_dim=time_emb_dim, hidden_dim=hidden_dim, num_heads=num_heads, depth=depth, 
            num_tokens=num_tokens, encoded_coord_dim=fourier_dim*2, use_node_type=use_node_type, 
            node_type_dim=node_type_dim
        )
        
        self.decoder = AttentionRenderer(
            latent_dim=latent_dim, coord_dim=coord_dim, t_chunk=t_chunk, channel_out=channel_out, 
            hidden_dim=hidden_dim, num_layers=num_layers, use_node_type=use_node_type, 
            node_type_dim=node_type_dim, encoded_coord_dim=fourier_dim*2, num_heads=num_heads
        )
        print("Using Attention Renderer")

    def forward(self, x_noisy, t, input_coords, query_coords, input_node_type=None, query_node_type=None):
        # 1. Encode coordinates
        input_coords_encoded = self.coord_encoder(input_coords)
        query_coords_encoded = self.coord_encoder(query_coords)
        
        # 2. Embed node types
        input_type_embeds = None
        query_type_embeds = None
        if self.use_node_type:
            if input_node_type is not None:
                input_type_embeds = self.node_type_embed(input_node_type.squeeze(-1).long())
            if query_node_type is not None:
                query_type_embeds = self.node_type_embed(query_node_type.squeeze(-1).long())
                
        # 3. Encoder extracts latent
        z_multi = self.encoder(x_noisy, input_coords_encoded, t, input_type_embeds)
        
        # 4. Decoder renders predictions
        out = self.decoder(z_multi, query_coords, query_coords_encoded, query_type_embeds)
        
        return out


