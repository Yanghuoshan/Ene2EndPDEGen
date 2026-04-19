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
        time = time * 1000.0  # Scale time to a larger range for better frequency coverage
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


class HyperNetwork_Perceiver_v22(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    When t_chunk is 128, the prediction is worse than t_chunk=16, likely due to the increased difficulty of learning stable Fourier features and attention over longer sequences.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=8, depth=4, num_tokens=16, fourier_dim=64, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk

        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        self.use_node_type = use_node_type

        self.node_type_num = 10 if use_node_type else 0

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
            nn.Linear(hidden_dim + encoded_coord_dim + self.node_type_num, hidden_dim),
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

    def forward(self, x_noisy, coords, t, node_type=None):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Build node features from trajectory and Fourier-encoded coordinates.
        x_freq = torch.fft.rfft(x_noisy, dim=1)  # [B, F, N, C]
        x_freq_real = torch.view_as_real(x_freq).permute(0, 2, 1, 3, 4).reshape(B, N, -1)  # [B, N, F * C * 2]
        freq_features = self.freq_proj(x_freq_real) # [B, N, hidden_dim]
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
        node_features = torch.cat([freq_features, coords_encoded], dim=-1)
        if self.use_node_type and node_type is not None:
            node_type = node_type.squeeze(-1) # [B, N, 1] -> [B, N]
            node_type_emb = F.one_hot(node_type, num_classes=self.node_type_num).float() # [B, N, num_node_types]
            node_features = torch.cat([node_features, node_type_emb], dim=-1)
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


class GaborRenderer_v22(nn.Module):
    """
    Direct faithful translation of GNAutodecoder_film from the official ConditionalNeuralField repo.
    Uses an MFN (Multiplicative Filter Network) structure with Gabor layers processing pure geometry 
    and Linear layers processing the features, modulated by the Latent target Z1 via additive shifting.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_node_type = use_node_type
        self.node_type_num = 10 if use_node_type else 0
        
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
            nn.Linear(coord_dim + self.node_type_num, hidden_dim),
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

    def forward(self, z_multi, coords, node_type=None):
        B, N, _ = coords.shape
        x0 = coords # x0 is [B, N, coord_dim], used for computing geometry basis in filters and as queries in cross-attention
        
        # Extract location-specific latent via Cross-Attention
        q = x0
        if self.use_node_type and node_type is not None:
            node_type = node_type.squeeze(-1) # [B, N, 1] -> [B, N]
            node_type_emb = F.one_hot(node_type, num_classes=self.node_type_num).float() # [B, N, self.node_type_num]
            q = torch.cat([x0, node_type_emb], dim=-1) # [B, N, coord_dim + self.node_type_num]
        q = self.query_proj(q)
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

        # Convert frequency domain back to time domain
        out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1) # [B, T_chunk, N, channel_out]
        return out




class TemporalSWT(nn.Module):
    """
    时间维度的多级平稳小波变换 (Stationary Wavelet Transform, 无下采样)
    输入: [B, T, N, C]
    输出: 一个包含 (levels + 1) 个张量的列表，每个张量的形状均为 [B, T, N, C]
          列表顺序为: [近似分量(低频), 细节分量_Level_L, ..., 细节分量_Level_1]
    """
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        # Haar 小波的低通和高通滤波器 (归一化为 1/sqrt(2))
        h0 = [0.7071067811865476, 0.7071067811865476]
        h1 = [-0.7071067811865476, 0.7071067811865476]
        
        self.register_buffer('h0', torch.tensor(h0, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('h1', torch.tensor(h1, dtype=torch.float32).view(1, 1, -1))

    def forward(self, x):
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        # 重排并合并维度以使用 1D 卷积: [B*N*C, 1, T]
        x = x.permute(0, 2, 3, 1).reshape(B * N * C, 1, T)
        
        results = []
        curr_x = x
        
        for level in range(self.levels):
            # 在平稳小波变换中，每一级不进行下采样，而是将滤波器的膨胀(dilation)翻倍
            dilation = 2 ** level
            # 使用反射填充(Reflection padding)来处理边界，防止信号长度改变
            # 对于长度为2的滤波器，膨胀为dilation，在左侧填充 dilation，右侧不填充（或根据需要对齐）
            curr_x_padded = F.pad(curr_x, (dilation, 0), mode='reflect')
            
            # 低通 (Approximation)
            low = F.conv1d(curr_x_padded, self.h0, dilation=dilation)
            # 高通 (Detail)
            high = F.conv1d(curr_x_padded, self.h1, dilation=dilation)
            
            # 保存高频细节特征到结果列表
            results.append(high.view(B, N, C * T))
            curr_x = low
            
        # 循环结束，保存最后的低频近似特征
        results.append(curr_x.view(B, N, C * T))
        
        # 反转列表，使得排列顺序为: [低频cA_L, 高频cD_L, 高频cD_L-1, ..., 高频cD_1]
        results.reverse()
        return results


class TemporalISWT(nn.Module):
    """
    时间维度的多级平稳逆小波变换 (Inverse Stationary Wavelet Transform)
    输入: coeffs, 一个包含 (levels + 1) 个张量的列表，每个张量的形状均为 [B, N, C, T]
          列表顺序需与 TemporalSWT 输出一致: [低频cA_L, 高频cD_L, 高频cD_L-1, ..., 高频cD_1]
    输出: [B, T, N, C]
    """
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        # Haar 逆小波重建滤波器 (归一化为 1/sqrt(2))
        g0 = [0.7071067811865476, 0.7071067811865476]
        g1 = [0.7071067811865476, -0.7071067811865476]
        
        # 注意：ConvTranspose1d的权重形状为 [in_channels, out_channels/groups, kernel_size]
        self.register_buffer('g0', torch.tensor(g0, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('g1', torch.tensor(g1, dtype=torch.float32).view(1, 1, -1))

    def forward(self, coeffs):
        # 提取出最低频的近似信号
        approx = coeffs[0]
        B, N, C, T = approx.shape
        # 转为 [B*N*C, 1, T] 准备 1D 转置卷积
        curr_x = approx.view(B * N * C, 1, T)
        
        # 细节信号列表，从最深层（粗糙）到最浅层（精细）
        details = coeffs[1:]
        
        for level in reversed(range(self.levels)):
            dilation = 2 ** level
            detail = details[self.levels - 1 - level]
            detail = detail.view(B * N * C, 1, T)
            
            # 为了抵消前向变换的偏移，我们需要在进行转置卷积后再截取正确的部分
            # 转置卷积 (相当于重建滤波和上采样重叠相加)
            low_rec = F.conv_transpose1d(curr_x, self.g0, dilation=dilation)
            high_rec = F.conv_transpose1d(detail, self.g1, dilation=dilation)
            
            # 合并低通和高通，并取平均以维持无下采样分解(SWT/MODWT)的能量守恒
            # 这里前向填充在了左侧 dilation，转置卷积会在右侧多出 dilation 个点
            merged = (low_rec + high_rec) / 2.0
            
            # 去除右侧多余的 padding
            curr_x = merged[..., :-dilation]
            
        # 还原回原始形状 [B, T, N, C]
        out = curr_x.view(B, N, C, T).permute(0, 3, 1, 2)
        return out


class HyperNetwork_Perceiver_v23(nn.Module):
    """
    DiT-style Set Encoder (Data-Space to Latent Z1)
    Uses Perceiver-IO styled Cross-Attention to query high-dimensional dynamics from N points,
    followed by DiT Self-Attention blocks conditioned on diffusion time t.
    """
    def __init__(self, t_chunk=16, channel_in=2, coord_dim=2, latent_dim=256, time_emb_dim=256, hidden_dim=256, num_heads=16, depth=4, num_tokens=16, fourier_dim=64, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk

        self.channel_in = channel_in
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim
        self.use_node_type = use_node_type

        self.node_type_num = 10 if use_node_type else 0

        self.coord_encoder = FourierFeatures(in_features=coord_dim, mapping_size=fourier_dim)
        encoded_coord_dim = fourier_dim * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Convert time domain [B, T, N, C] to wavelet domain
        self.swt_levels = 3
        self.swt = TemporalSWT(levels=self.swt_levels)
        freq_dim = (self.swt_levels + 1) * t_chunk * channel_in 
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Maps temporal features + encoded coords to node features.
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim + encoded_coord_dim + self.node_type_num, hidden_dim),
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

    def forward(self, x_noisy, coords, t, node_type=None):
        B, T, N, C = x_noisy.shape
        
        # 1. Condition from time
        c = self.time_mlp(t) # [B, hidden_dim]
        
        # 2. Build node features from trajectory and Fourier-encoded coordinates.
        swt_coeffs = self.swt(x_noisy) # List of 4 tensors of shape [B, N, C * T]
        x_swt = torch.cat(swt_coeffs, dim=-1) # [B, N, (levels+1)*T*C]
        
        freq_features = self.freq_proj(x_swt) # [B, N, hidden_dim]
        coords_encoded = self.coord_encoder(coords) # [B, N, fourier_dim * 2]
        node_features = torch.cat([freq_features, coords_encoded], dim=-1)
        if self.use_node_type and node_type is not None:
            node_type = node_type.squeeze(-1) # [B, N, 1] -> [B, N]
            node_type_emb = F.one_hot(node_type, num_classes=self.node_type_num).float() # [B, N, num_node_types]
            node_features = torch.cat([node_features, node_type_emb], dim=-1)
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



class MFNBranch(nn.Module):
    def __init__(self, latent_dim, coord_dim, hidden_dim, num_layers, out_dim, freq_scale=1.0):
        super().__init__()
        self.filters = nn.ModuleList([
            FullGaborLayer(
                in_features=coord_dim, 
                out_features=hidden_dim, 
                weight_scale=(256.0 * freq_scale) / math.sqrt(num_layers + 1), 
                alpha=6.0 / (num_layers + 1), 
                beta=1.0
            ) 
            for _ in range(num_layers + 1)
        ])
        
        self.net1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.net2 = nn.ModuleList([nn.Linear(latent_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)])
        self.final_linear = nn.Linear(hidden_dim, out_dim)

        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(-math.sqrt(1.0 / in_dim), math.sqrt(1.0 / in_dim))
            
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(-math.sqrt(1.0 / in_dim), math.sqrt(1.0 / in_dim))

    def forward(self, z, x0):
        x = self.filters[0](x0) * self.net2[0](z)
        for i in range(1, len(self.filters)):
            basis = self.filters[i](x0)
            h = self.net1[i - 1](x) + self.net2[i](z)
            x = basis * h
        return self.final_linear(x)


class GaborRenderer_v23(nn.Module):
    """
    Wavelet-based GaborRenderer using 4 separate Multiplicative Filter Networks.
    Generates 4 temporal SWT subbands from lowest to highest frequency with conditional cascade.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, use_node_type=False):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_node_type = use_node_type
        self.node_type_num = 10 if use_node_type else 0

        # Cross-attention to extract coordinate-specific latents
        self.query_proj = nn.Sequential(
            nn.Linear(coord_dim + self.node_type_num, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=16, batch_first=True)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        # 4 branches for Temporal SWT (levels=3 produces 4 bands: cA_3, cD_3, cD_2, cD_1)
        out_dim = t_chunk * channel_out
        
        # We assign different freq_scale initializations to handle different frequency patterns
        self.branch_cA3 = MFNBranch(latent_dim, coord_dim, hidden_dim, num_layers, out_dim, freq_scale=0.5)  # Lowest freq
        self.branch_cD3 = MFNBranch(latent_dim, coord_dim, hidden_dim, num_layers, out_dim, freq_scale=1.0)
        self.branch_cD2 = MFNBranch(latent_dim, coord_dim, hidden_dim, num_layers, out_dim, freq_scale=2.0)
        self.branch_cD1 = MFNBranch(latent_dim, coord_dim, hidden_dim, num_layers, out_dim, freq_scale=4.0)  # Highest freq

        # Conditioning pathway: map low-frequency generation to condition subsequent branches
        self.cond_projA3 = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        nn.init.zeros_(self.cond_projA3[-1].weight)
        if self.cond_projA3[-1].bias is not None:
            nn.init.zeros_(self.cond_projA3[-1].bias)

        self.cond_projD3 = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        nn.init.zeros_(self.cond_projD3[-1].weight)
        if self.cond_projD3[-1].bias is not None:
            nn.init.zeros_(self.cond_projD3[-1].bias)

        self.cond_projD2 = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        nn.init.zeros_(self.cond_projD2[-1].weight)
        if self.cond_projD2[-1].bias is not None:
            nn.init.zeros_(self.cond_projD2[-1].bias)

        # Inverse SWT mapped to decode 4 subbands back to full temporal space
        self.iswt = TemporalISWT(levels=3)

    def forward(self, z_multi, coords, node_type=None):
        B, N, _ = coords.shape
        x0 = coords # x0 is [B, N, coord_dim]
        
        # Extract location-specific latent via Cross-Attention
        q = x0
        if self.use_node_type and node_type is not None:
            node_type = node_type.squeeze(-1) # [B, N, 1] -> [B, N]
            node_type_emb = F.one_hot(node_type, num_classes=self.node_type_num).float()
            q = torch.cat([x0, node_type_emb], dim=-1)
        q = self.query_proj(q)
        q = self.norm_q(q)
        kv = self.norm_kv(z_multi)
        z, _ = self.cross_attn(q, kv, kv) # z is [B, N, latent_dim]
        
        # 1. Generate lowest frequency band first
        out_cA3 = self.branch_cA3(z, x0)  # [B, N, t_chunk * channel_out]
        
        # 2. Conditionally generate remaining bands using low-frequency as conditioning
        z_cond = z + self.cond_projA3(out_cA3)
        out_cD3 = self.branch_cD3(z_cond, x0)

        z_cond = z + self.cond_projD3(out_cD3)
        out_cD2 = self.branch_cD2(z_cond, x0)

        z_cond = z + self.cond_projD2(out_cD2)
        out_cD1 = self.branch_cD1(z_cond, x0)
        
        def reshape_band(out_band):
            # Transform [B, N, T * C] to [B, N, C, T]
            return out_band.view(B, N, self.channel_out, self.t_chunk)
            
        band_cA3 = reshape_band(out_cA3)
        band_cD3 = reshape_band(out_cD3)
        band_cD2 = reshape_band(out_cD2)
        band_cD1 = reshape_band(out_cD1)
        
        # 3. Inverse Stationary Wavelet Transform
        coeffs = [band_cA3, band_cD3, band_cD2, band_cD1]
        out = self.iswt(coeffs)  
        
        return out
