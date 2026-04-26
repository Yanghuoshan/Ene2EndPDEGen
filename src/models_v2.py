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
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64, use_freq_filter=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.use_freq_filter = use_freq_filter
        if self.use_freq_filter:
            self.freq_weight = nn.Parameter(torch.ones(1, t_chunk // 2 + 1, 1, 1))
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
        
        # Convert time domain [B, T, N, C] to frequency domain
        freq_dim = (t_chunk // 2 + 1) * 2 * channel_in # t_chunk = 64 -> freq_dim = 33 * 2 * 2 = 132
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps temporal features + encoded coords to node features.
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + encoded_coord_dim, hidden_dim),
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

    def forward(self, x_noisy, coords, t):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Build node features from trajectory and Fourier-encoded coordinates.
        x_freq = torch.fft.rfft(x_noisy, dim=1)  # [B, F, N, C]
        if self.use_freq_filter:
            x_freq = x_freq * self.freq_weight
        x_freq_real = torch.view_as_real(x_freq).reshape(B, N, -1)  # [B, N, F * C * 2]
        freq_features = self.freq_proj(x_freq_real) # [B, N, hidden_dim]
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
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
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, use_freq_filter=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.use_freq_filter = use_freq_filter
        if self.use_freq_filter:
            self.freq_weight = nn.Parameter(torch.ones(1, t_chunk // 2 + 1, 1, 1))
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
        self.query_proj = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
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

    def forward(self, z_multi, coords):
        B, N, _ = coords.shape
        x0 = coords # x0 is [B, N, coord_dim], used for computing geometry basis in filters and as queries in cross-attention
        
        # Extract location-specific latent via Cross-Attention
        q = self.query_proj(x0)
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
        out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4)
        out_freq_complex = torch.view_as_complex(out_freq_real) # [B, F, N, channel_out]
        
        if self.use_freq_filter:
            out_freq_complex = out_freq_complex * self.freq_weight

        # Convert frequency domain back to time domain
        out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1) # [B, T_chunk, N, channel_out]
        return out


class FNOLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        初始化1D FNO层。
        :param in_channels: 输入特征/通道数
        :param out_channels: 输出特征/通道数
        :param modes: 保留的低频傅里叶模式数，应满足 modes <= T // 2 + 1
        """
        super(FNOLayer1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # 局部线性变换 W * u_l，适用于维度 [B，C，T]
        self.w = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # 谱域变换复数权重 R
        # 缩放因子有助于训练稳定
        scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        """
        复数乘法。
        input shape: [Batch * N, in_channels, modes]
        weights shape: [in_channels, out_channels, modes]
        output shape: [Batch * N, out_channels, modes]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, u):
        """
        前向传播。
        u shape: [Batch * N, Channels, T]
        """
        batchsize = u.shape[0]
        T_len = u.shape[-1]
        
        # 1. 计算 F(u_l)
        # 对最后一个维度(T)计算 rfft，输出形如 [Batch*N, Channels, T//2 + 1]
        u_ft = torch.fft.rfft(u, norm="ortho")

        # 2. 计算 R(F(u_l))
        # 构造跟傅里叶特征相同尺寸的新复数占位张量，初始化为 0（代表截断高频）
        out_ft = torch.zeros(batchsize, self.out_channels, u_ft.size(-1), dtype=torch.cfloat, device=u.device)
        
        # 截取前 self.modes 个低频模态并完成复数矩阵乘积作为特征更新
        out_ft[:, :, :self.modes] = self.compl_mul1d(u_ft[:, :, :self.modes], self.weights)

        # 3. 计算 F^{-1}(...) 回到时域/物理域
        # 指定 n=T_len 确保由于奇偶可能带来的序列长度问题准确对齐原本周期
        u_fourier = torch.fft.irfft(out_ft, n=T_len, norm="ortho")
        
        # 4. 计算 W * u_l
        u_local = self.w(u)
        
        # 5. 返回两部分和 u_{l+1}
        return u_fourier + u_local



class HyperNetwork_Perceiver_v3(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1) with FNOLayer1D at the beginning.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64, fno_modes=8):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim

        # FNO Layer at the beginning mapping channel_in to channel_in
        self.fno = FNOLayer1D(in_channels=channel_in, out_channels=channel_in, modes=fno_modes)

        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
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
        
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))

        self.cross_blocks = nn.ModuleList([
            CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.self_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x_noisy, coords, t):
        """
        x_noisy shape: [B, T, N, C]
        coords shape: [B, N, coord_dim]
        """
        B, T, N, C = x_noisy.shape
        
        # --- FNOLayer1D Pre-processing ---
        # Reshape from [B, T, N, C] to [Batch * N, C, T] for FNO layer
        x_fno = x_noisy.permute(0, 2, 3, 1).reshape(B * N, C, T)
        x_fno = self.fno(x_fno)
        # Reshape back to [B, T, N, C]
        x_noisy = x_fno.view(B, N, C, T).permute(0, 3, 1, 2)
        # ---------------------------------
        
        c = self.time_mlp(t)
        
        x_freq = torch.fft.rfft(x_noisy, dim=1, norm="ortho")
        x_freq_real = torch.view_as_real(x_freq).reshape(B, N, -1)
        freq_features = self.freq_proj(x_freq_real)
        coords_encoded = self.coord_encoder(coords)
        node_features = torch.cat([freq_features, coords_encoded], dim=-1)
        feat = self.node_proj(node_features)
        
        q = self.query_tokens.expand(B, -1, -1)
        latents = q
        
        for cross_block, self_block in zip(self.cross_blocks, self.self_blocks):
            latents = cross_block(latents, feat, c)
            latents = self_block(latents, c)
            
        z_multi = self.final_proj(latents)
        
        return z_multi


class GaborRenderer_v3(nn.Module):
    """
    Direct faithful translation of GNAutodecoder_film from the official ConditionalNeuralField repo.
    Uses MFN structure with Gabor layers processing pure geometry and includes an FNOLayer1D at the end.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, fno_modes=8):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
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
        
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        
        self.net2 = nn.ModuleList(
            [nn.Linear(latent_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)]
        )

        self.query_proj = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        self.freq_dim = (t_chunk // 2 + 1)
        self.final_linear = nn.Linear(hidden_dim, self.freq_dim * 2 * channel_out)
        
        # FNO Layer at the end mapping channel_out to channel_out
        self.fno = FNOLayer1D(in_channels=channel_out, out_channels=channel_out, modes=fno_modes)
        
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim),
            )
            
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim)
            )

    def forward(self, z_multi, coords):
        B, N, _ = coords.shape
        x0 = coords
        
        q = self.query_proj(x0)
        q = self.norm_q(q)
        kv = self.norm_kv(z_multi)
        z, _ = self.cross_attn(q, kv, kv)
        
        x = self.filters[0](x0) * self.net2[0](z)
        
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0)
            h = self.net1[i - 1](x) + self.net2[i](z)
            x = basis * h
            
        out = self.final_linear(x)
        
        out_freq_real = out.view(B, N, self.freq_dim, self.channel_out, 2)
        out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4)
        out_freq_complex = torch.view_as_complex(out_freq_real)
        

        out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        
        # --- FNOLayer1D Post-processing ---
        # Reshape from [B, T_chunk, N, channel_out] to [Batch * N, channel_out, T_chunk] for FNO layer
        out_fno = out.permute(0, 2, 3, 1).reshape(B * N, self.channel_out, self.t_chunk)
        out_fno = self.fno(out_fno)
        
        # Apply frequency truncation (keep at most 32 frequencies)
        out_fno_freq = torch.fft.rfft(out_fno, dim=-1, norm="ortho")
        if out_fno_freq.size(-1) > 32:
            out_fno_freq[..., 32:] = 0
        out_fno = torch.fft.irfft(out_fno_freq, n=self.t_chunk, dim=-1, norm="ortho")
        
        # Reshape back to [B, T_chunk, N, channel_out]
        out = out_fno.view(B, N, self.channel_out, self.t_chunk).permute(0, 3, 1, 2)
        # ----------------------------------
        
        return out


class ComplexMixingLayer(nn.Module):
    def __init__(self, num_freqs, num_channels):
        """
        :param num_freqs: F所在维度的长度
        :param num_channels: channel_out所在维度的长度
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.num_channels = num_channels
        
        # 频率混合权重，shape: [F, F]
        scale_f = 1.0 / math.sqrt(num_freqs)
        self.freq_weight = nn.Parameter(scale_f * torch.randn(num_freqs, num_freqs, dtype=torch.cfloat))
        self.freq_bias = nn.Parameter(torch.zeros(num_freqs, dtype=torch.cfloat))
        
        # 通道混合权重，shape: [C, C]
        scale_c = 1.0 / math.sqrt(num_channels)
        self.channel_weight = nn.Parameter(scale_c * torch.randn(num_channels, num_channels, dtype=torch.cfloat))
        self.channel_bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.cfloat))

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，shape [B, F, N, C], 需要是复数类型 (如 torch.cfloat)
        :return: 输出张量，shape [B, F, N, C]
        """
        # 1. 频率混合 (Frequency mixing)
        # b: batch, f: in_freq, n: node, i: channel, g: out_freq
        x = torch.einsum("bfni,fg->bgni", x, self.freq_weight)
        # 加上 freq 维度的偏置，利用广播机制
        x = x + self.freq_bias.view(1, -1, 1, 1)
        
        # 2. 通道混合 (Channel mixing)
        # b: batch, f: freq, n: node, i: in_channel, j: out_channel
        x = torch.einsum("bfni,ij->bfnj", x, self.channel_weight)
        # 加上 channel 维度的偏置
        x = x + self.channel_bias.view(1, 1, 1, -1)
        
        return x


class HyperNetwork_Perceiver_v4(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1).
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64, fno_modes=8, use_fft=False, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        self.depth = depth
        self.use_fft = use_fft
        self.use_node_type = use_node_type
        self.node_type_dim = 10 if use_node_type else 0

        self.coord_encoder = FourierFeatures(in_features=coord_dim + self.node_type_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if self.use_fft:
            freq_dim = (t_chunk // 2 + 1) * 2 * channel_in
        else:
            freq_dim = t_chunk * channel_in
            
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Pre-process node features with self-attention before cross-attending from latents.
        # self.feat_self_block = DiTBlock(hidden_dim, num_heads)
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))

        self.cross_blocks = nn.ModuleList([
            CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.self_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.feat_cross = nn.ModuleList([
            CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth - 1)  # 最后一层不进行特征交叉
        ])
        
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x_noisy, coords, t, node_type=None):
        """
        x_noisy shape: [B, T, N, C]
        coords shape: [B, N, coord_dim]
        """
        if self.use_node_type and node_type is not None:
            node_type_onehot = F.one_hot(node_type.squeeze(-1).long(), num_classes=10).float()
            coords = torch.cat([coords, node_type_onehot], dim=-1)

        B, T, N, C = x_noisy.shape
        
        c = self.time_mlp(t)
        
        if self.use_fft:
            x_freq = torch.fft.rfft(x_noisy, dim=1, norm="ortho") # [B, F, N, C]
            x_freq_real = torch.view_as_real(x_freq).reshape(B, N, -1) # [B, N, F * C * 2]
            freq_features = self.freq_proj(x_freq_real)
        else:
            x_time_flat = x_noisy.permute(0, 2, 1, 3).reshape(B, N, -1)
            freq_features = self.freq_proj(x_time_flat)
            
        coords_encoded = self.coord_encoder(coords)
        node_features = torch.cat([freq_features, coords_encoded], dim=-1)
        feat = self.node_proj(node_features)
        # feat = self.feat_self_block(feat, c)
        
        q = self.query_tokens.expand(B, -1, -1)
        latents = q
        
        for i in range(self.depth):
            latents = self.cross_blocks[i](latents, feat, c)
            latents = self.self_blocks[i](latents, c)
            if i < self.depth - 1:   # 不是最后一层
                feat = feat + 0.1 * self.feat_cross[i](feat, latents, c)
            
        z_multi = self.final_proj(latents)
        
        return z_multi


class GaborRenderer_v4(nn.Module):
    """
    Direct faithful translation of GNAutodecoder_film from the official ConditionalNeuralField repo.
    Uses MFN structure with Gabor layers processing pure geometry.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, fno_modes=8, use_fft=False, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_fft = use_fft
        self.use_node_type = use_node_type
        self.node_type_dim = 10 if use_node_type else 0
        
        self.filters = nn.ModuleList([
            FullGaborLayer(
                in_features=coord_dim + self.node_type_dim, 
                out_features=hidden_dim, 
                weight_scale=256.0 / math.sqrt(num_layers + 1), 
                alpha=6.0 / (num_layers + 1), 
                beta=1.0
            ) 
            for _ in range(num_layers + 1)
        ])
        
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        
        self.net2 = nn.ModuleList(
            [nn.Linear(latent_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)]
        )

        self.query_proj = nn.Sequential(
            nn.Linear(coord_dim + self.node_type_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        if self.use_fft:
            self.freq_dim = (t_chunk // 2 + 1)
            out_dim = self.freq_dim * 2 * channel_out
        else:
            out_dim = t_chunk * channel_out
            
        self.final_linear = nn.Linear(hidden_dim, out_dim)
        
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim),
            )
            
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim)
            )

    def forward(self, z_multi, coords, node_type=None):
        if self.use_node_type and node_type is not None:
            node_type_onehot = F.one_hot(node_type.squeeze(-1).long(), num_classes=10).float()
            coords = torch.cat([coords, node_type_onehot], dim=-1)

        B, N, _ = coords.shape
        x0 = coords
        
        q = self.query_proj(x0)
        q = self.norm_q(q)
        kv = self.norm_kv(z_multi)
        z, _ = self.cross_attn(q, kv, kv)
        
        x = self.filters[0](x0) * self.net2[0](z)
        
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0)
            h = self.net1[i - 1](x) + self.net2[i](z)
            x = basis * h
            
        out = self.final_linear(x)
        
        if self.use_fft:
            out_freq_real = out.view(B, N, self.freq_dim, self.channel_out, 2)
            out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4)
            out_freq_complex = torch.view_as_complex(out_freq_real)
            out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        else:
            # Reshape back to [B, T_chunk, N, channel_out]
            out = out.view(B, N, self.t_chunk, self.channel_out).permute(0, 2, 1, 3)

        # 后处理：频率截断

        out_freq = torch.fft.rfft(out, dim=1, norm="ortho") # [B, F, N, channel_out]
        
        # Truncation
        out_freq_trunc = out_freq.clone()
        if out_freq_trunc.size(1) > 32:
            out_freq_trunc[:, 32:, ...] = 0
            
        out = torch.fft.irfft(out_freq_trunc, n=self.t_chunk, dim=1, norm="ortho")
        
        return out


class HyperNetwork_Perceiver_v5(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1).
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64, fno_modes=8, use_fft=True, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        self.depth = depth
        self.use_fft = use_fft
        self.use_node_type = use_node_type
        self.node_type_dim = 10 if use_node_type else 0

        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim + self.node_type_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        encoded_coord_dim = latent_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if self.use_fft:
            freq_dim = (t_chunk // 2 + 1) * 2 * channel_in
        else:
            freq_dim = t_chunk * channel_in
            
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Pre-process node features with self-attention before cross-attending from latents.
        # self.feat_self_block = DiTBlock(hidden_dim, num_heads)
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))

        self.cross_blocks = nn.ModuleList([
            CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.self_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        # self.feat_cross = nn.ModuleList([
        #     CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth - 1)  # 最后一层不进行特征交叉
        # ])
        
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x_noisy, coords, t, node_type=None):
        """
        x_noisy shape: [B, T, N, C]
        coords shape: [B, N, coord_dim]
        """
        if self.use_node_type and node_type is not None:
            node_type_onehot = F.one_hot(node_type.squeeze(-1).long(), num_classes=10).float()
            coords = torch.cat([coords, node_type_onehot], dim=-1)

        B, T, N, C = x_noisy.shape
        
        c = self.time_mlp(t)
        
        if self.use_fft:
            x_freq = torch.fft.rfft(x_noisy, dim=1, norm="ortho") # [B, F, N, C]
            x_freq_real = torch.view_as_real(x_freq).reshape(B, N, -1) # [B, N, F * C * 2]
            freq_features = self.freq_proj(x_freq_real)
        else:
            x_time_flat = x_noisy.permute(0, 2, 1, 3).reshape(B, N, -1)
            freq_features = self.freq_proj(x_time_flat)
            
        coords_encoded = self.coord_encoder(coords)
        node_features = torch.cat([freq_features, coords_encoded], dim=-1)
        feat = self.node_proj(node_features)
        # feat = self.feat_self_block(feat, c)
        
        q = self.query_tokens.expand(B, -1, -1)
        latents = q
        
        for i in range(self.depth):
            latents = self.cross_blocks[i](latents, feat, c)
            latents = self.self_blocks[i](latents, c)
            # if i < self.depth - 1:   # 不是最后一层
            #     feat = feat + 0.1 * self.feat_cross[i](feat, latents, c)
            
        z_multi = self.final_proj(latents)
        
        return z_multi


class GaborRenderer_v5(nn.Module):
    """
    Direct faithful translation of GNAutodecoder_film from the official ConditionalNeuralField repo.
    Uses MFN structure with Gabor layers processing pure geometry.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, fno_modes=8, use_fft=True, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_fft = use_fft
        self.use_node_type = use_node_type
        self.node_type_dim = 10 if use_node_type else 0
        
        self.filters = nn.ModuleList([
            FullGaborLayer(
                in_features=coord_dim + self.node_type_dim, 
                out_features=hidden_dim, 
                weight_scale=256.0 / math.sqrt(num_layers + 1), 
                alpha=6.0 / (num_layers + 1), 
                beta=1.0
            ) 
            for _ in range(num_layers + 1)
        ])
        
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        
        self.net2 = nn.ModuleList(
            [nn.Linear(latent_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)]
        )

        self.query_proj = nn.Sequential(
            nn.Linear(coord_dim + self.node_type_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        if self.use_fft:
            self.freq_dim = (t_chunk // 2 + 1)
            out_dim = self.freq_dim * 2 * channel_out
        else:
            out_dim = t_chunk * channel_out
            
        self.final_linear = nn.Linear(hidden_dim, out_dim)
        
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim),
            )
            
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim)
            )

    def forward(self, z_multi, coords, node_type=None):
        if self.use_node_type and node_type is not None: # [B, N, 1]
            node_type_onehot = F.one_hot(node_type.squeeze(-1).long(), num_classes=10).float()
            coords = torch.cat([coords, node_type_onehot], dim=-1)

        B, N, _ = coords.shape
        x0 = coords
        
        q = self.query_proj(x0)
        q = self.norm_q(q)
        kv = self.norm_kv(z_multi)
        z, _ = self.cross_attn(q, kv, kv)
        
        x = self.filters[0](x0) * self.net2[0](z)
        
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0)
            h = self.net1[i - 1](x) + self.net2[i](z)
            x = basis * h
            
        out = self.final_linear(x)
        
        if self.use_fft:
            out_freq_real = out.view(B, N, self.freq_dim, self.channel_out, 2)
            out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4)
            out_freq_complex = torch.view_as_complex(out_freq_real)
            out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        else:
            # Reshape back to [B, T_chunk, N, channel_out]
            out = out.view(B, N, self.t_chunk, self.channel_out).permute(0, 2, 1, 3)

        # 后处理：频率截断

        # out_freq = torch.fft.rfft(out, dim=1, norm="ortho") # [B, F, N, channel_out]
        
        # Truncation
        # out_freq_trunc = out_freq.clone()
        # if out_freq_trunc.size(1) > 40:
        #     out_freq_trunc[:, 40:, ...] = 0
            
        # out = torch.fft.irfft(out_freq_trunc, n=self.t_chunk, dim=1, norm="ortho")
        
        return out


class HyperNetwork_Perceiver_v55(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1).
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64, fno_modes=8, use_fft=True, use_node_type=False, num_node_types=10, node_type_emb_dim=32):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        self.depth = depth
        self.use_fft = use_fft
        self.use_node_type = use_node_type
        self.num_node_types = num_node_types
        self.node_type_emb_dim = node_type_emb_dim
        self.node_type_dim = node_type_emb_dim if use_node_type else 0

        if self.use_node_type:
            self.node_type_embedding = nn.Embedding(num_node_types, node_type_emb_dim)
            self.node_type_feat_proj = nn.Sequential(
                nn.Linear(node_type_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.node_type_query_kv = nn.Sequential(
                nn.Linear(node_type_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.node_type_query_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim + self.node_type_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        encoded_coord_dim = latent_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if self.use_fft:
            freq_dim = (t_chunk // 2 + 1) * 2 * channel_in
        else:
            freq_dim = t_chunk * channel_in
            
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + encoded_coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Pre-process node features with self-attention before cross-attending from latents.
        # self.feat_self_block = DiTBlock(hidden_dim, num_heads)
        
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)/math.sqrt(hidden_dim))

        self.cross_blocks = nn.ModuleList([
            CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.self_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        # self.feat_cross = nn.ModuleList([
        #     CrossDiTBlock(hidden_dim, num_heads) for _ in range(depth - 1)  # 最后一层不进行特征交叉
        # ])
        
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x_noisy, coords, t, node_type=None):
        """
        x_noisy shape: [B, T, N, C]
        coords shape: [B, N, coord_dim]
        """
        B, T, N, C = x_noisy.shape

        node_type_embed = None
        if self.use_node_type:
            if node_type is None:
                node_type_embed = coords.new_zeros(B, N, self.node_type_emb_dim)
            else:
                node_type_ids = node_type.squeeze(-1).long().clamp(0, self.num_node_types - 1)
                node_type_embed = self.node_type_embedding(node_type_ids)
            coords_input = torch.cat([coords, node_type_embed], dim=-1)
        else:
            coords_input = coords
        
        c = self.time_mlp(t)
        
        if self.use_fft:
            x_freq = torch.fft.rfft(x_noisy, dim=1, norm="ortho") # [B, F, N, C]
            x_freq_real = torch.view_as_real(x_freq).permute(0, 2, 1, 3, 4).reshape(B, N, -1) # [B, N, F * C * 2]
            freq_features = self.freq_proj(x_freq_real)
        else:
            x_time_flat = x_noisy.permute(0, 2, 1, 3).reshape(B, N, -1)
            freq_features = self.freq_proj(x_time_flat)
            
        coords_encoded = self.coord_encoder(coords_input)
        node_features = torch.cat([freq_features, coords_encoded], dim=-1)
        feat = self.node_proj(node_features)
        if self.use_node_type:
            feat = feat + self.node_type_feat_proj(node_type_embed)
        # feat = self.feat_self_block(feat, c)
        
        q = self.query_tokens.expand(B, -1, -1)
        if self.use_node_type:
            # Let each latent token read node-type context independently.
            type_kv = self.node_type_query_kv(node_type_embed)
            q_type_delta, _ = self.node_type_query_attn(q, type_kv, type_kv)
            q = q + q_type_delta
        latents = q
        
        for i in range(self.depth):
            latents = self.cross_blocks[i](latents, feat, c)
            latents = self.self_blocks[i](latents, c)
            # if i < self.depth - 1:   # 不是最后一层
            #     feat = feat + 0.1 * self.feat_cross[i](feat, latents, c)
            
        z_multi = self.final_proj(latents)
        
        return z_multi


class GaborRenderer_v55(nn.Module):
    """
    Direct faithful translation of GNAutodecoder_film from the official ConditionalNeuralField repo.
    Uses MFN structure with Gabor layers processing pure geometry.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, fno_modes=8, use_fft=True, use_node_type=False, num_node_types=10, node_type_emb_dim=32):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_fft = use_fft
        self.use_node_type = use_node_type
        self.num_node_types = num_node_types
        self.node_type_emb_dim = node_type_emb_dim
        self.node_type_dim = node_type_emb_dim if use_node_type else 0

        if self.use_node_type:
            self.node_type_embedding = nn.Embedding(num_node_types, node_type_emb_dim)
            self.node_type_z_mod = nn.Sequential(
                nn.Linear(node_type_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2 * latent_dim)
            )
        
        self.filters = nn.ModuleList([
            FullGaborLayer(
                in_features=coord_dim + self.node_type_dim, 
                out_features=hidden_dim, 
                weight_scale=256.0 / math.sqrt(num_layers + 1), 
                alpha=6.0 / (num_layers + 1), 
                beta=1.0
            ) 
            for _ in range(num_layers + 1)
        ])
        
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        
        self.net2 = nn.ModuleList(
            [nn.Linear(latent_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)]
        )

        self.query_proj = nn.Sequential(
            nn.Linear(coord_dim + self.node_type_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        if self.use_fft:
            self.freq_dim = (t_chunk // 2 + 1)
            out_dim = self.freq_dim * 2 * channel_out
        else:
            out_dim = t_chunk * channel_out
            
        self.final_linear = nn.Linear(hidden_dim, out_dim)
        
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim),
            )
            
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -math.sqrt(1.0 / in_dim),
                math.sqrt(1.0 / in_dim)
            )

    def forward(self, z_multi, coords, node_type=None):
        B, N, _ = coords.shape
        node_type_embed = None
        if self.use_node_type:
            if node_type is None:
                node_type_embed = coords.new_zeros(B, N, self.node_type_emb_dim)
            else:
                node_type_ids = node_type.squeeze(-1).long().clamp(0, self.num_node_types - 1)
                node_type_embed = self.node_type_embedding(node_type_ids)
            x0 = torch.cat([coords, node_type_embed], dim=-1)
        else:
            x0 = coords
        
        q = self.query_proj(x0)
        q = self.norm_q(q)
        kv = self.norm_kv(z_multi)
        z, _ = self.cross_attn(q, kv, kv)
        if self.use_node_type:
            z_scale, z_shift = self.node_type_z_mod(node_type_embed).chunk(2, dim=-1)
            z = z * (1.0 + z_scale) + z_shift
        
        x = self.filters[0](x0) * self.net2[0](z)
        
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0)
            h = self.net1[i - 1](x) + self.net2[i](z)
            x = basis * h
            
        out = self.final_linear(x)
        
        if self.use_fft:
            out_freq_real = out.view(B, N, self.freq_dim, self.channel_out, 2)
            out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4)
            out_freq_complex = torch.view_as_complex(out_freq_real)
            out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        else:
            # Reshape back to [B, T_chunk, N, channel_out]
            out = out.view(B, N, self.t_chunk, self.channel_out).permute(0, 2, 1, 3)

        # 后处理：频率截断

        # out_freq = torch.fft.rfft(out, dim=1, norm="ortho") # [B, F, N, channel_out]
        
        # Truncation
        # out_freq_trunc = out_freq.clone()
        # if out_freq_trunc.size(1) > 40:
        #     out_freq_trunc[:, 40:, ...] = 0
            
        # out = torch.fft.irfft(out_freq_trunc, n=self.t_chunk, dim=1, norm="ortho")
        
        return out