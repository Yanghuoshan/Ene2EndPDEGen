import time
import math
import torch
import torch.nn.functional as F


def display_current_data_time():
    """显示当前时间"""
    local_time = time.localtime()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    print(current_time)


def count_parameters(model, name="Model"):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in {name}: {total_params:,}")


def generate_spatial_grf(coords, target_shape, length_scale=0.2, grid_size=32):
    """
    Grid Interpolation-based Gaussian Random Field (GRF) Generator.
    利用频域生成平滑高斯随机场，并对离散(坐标)点进行双线性插值采样。
    
    参数:
    coords: [B, N, 2], 归一化到 [-1, 1] 范围的空间坐标
    target_shape: 期望的输出张量形状 [B, T, N, C]
    length_scale: 空间相关性尺度，越大越平滑
    grid_size: 频域生成的特征网格分辨率
    """
    B, T, N, C = target_shape
    device = coords.device
    
    # 1. 构造频域网格 (k_x, k_y)
    k_coord = torch.fft.fftfreq(grid_size) * grid_size  # [grid_size]
    kx, ky = torch.meshgrid(k_coord, k_coord, indexing='ij')
    k_sq = kx**2 + ky**2
    
    # 2. 构造功率谱 (高斯核)
    spectrum = torch.exp(-0.5 * k_sq * (length_scale * 2 * math.pi)**2)
    amplitude = torch.sqrt(spectrum).to(device) # [grid_size, grid_size]
    
    # 3. 采样复数标准白噪声
    noise_real = torch.randn(B, T * C, grid_size, grid_size, device=device)
    noise_imag = torch.randn(B, T * C, grid_size, grid_size, device=device)
    noise_complex = torch.complex(noise_real, noise_imag)
    
    # 4. 频域滤波并逆变换回空域
    filtered_complex = noise_complex * amplitude.unsqueeze(0).unsqueeze(0)
    grf_grid = torch.fft.ifft2(filtered_complex).real
    
    # 标准化，使得其在空间上保持近似标准正态的分布 (方差1, 均值0)
    grf_grid = (grf_grid - grf_grid.mean(dim=(2,3), keepdim=True)) / (grf_grid.std(dim=(2,3), keepdim=True) + 1e-5)
    
    # 5. 插值到离散点
    # grid_sample 要求采样网格形状为 [B, H_out, W_out, 2]
    # 我们把要采样的 N 个点看作 高度为1，宽度为N 的网格
    sample_coords = coords.unsqueeze(1) # [B, 1, N, 2]
    
    # 采样结果为 [B, T*C, 1, N]
    sampled_grf = F.grid_sample(grf_grid, sample_coords, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    # 6. Reshape 回目标形状 [B, T, N, C]
    sampled_grf = sampled_grf.squeeze(2) # [B, T*C, N]
    sampled_grf = sampled_grf.view(B, T, C, N).permute(0, 1, 3, 2).contiguous() # [B, T, N, C]
    
    return sampled_grf
