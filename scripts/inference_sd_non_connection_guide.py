import sys
import os
import glob
import torch
from basicutility import ReadInput as ri
from src.dataset import TrajectoryChunkDataset, H5DirectoryChunkDataset
from src.models import HyperNetwork, CNFRenderer
from src.models_ae import HyperNetwork_GINO, GaborRenderer_GINO
from src.models_v22 import HyperNetwork_Perceiver_v22, GaborRenderer_v22, HyperNetwork_Perceiver_v23, GaborRenderer_v23
from src.normalize import Normalizer_ts
from time import time

def inference_demo(hp):
    """
    Demonstrates how generation works after the model is fully trained.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 0. Configuration Settings
    DATASET_PATH = hp.dataset_path
    
    T_CHUNK = getattr(hp, "chunk_size", 16)
    C_OUT = getattr(hp, "c_out", 2)
    LATENT_DIM = getattr(hp, "latent_dim", 256)
    HIDDEN_DIM = getattr(hp, "hidden_dim", 256)
    DEPTH_ENC = getattr(hp, "depth_enc", 4)
    NUM_TOKENS = getattr(hp, "num_tokens", 16)
    NUM_LAYERS_CNF = getattr(hp, "num_layers_cnf", 4)
    STRIDE = getattr(hp, "stride", T_CHUNK)
    ENCODER_TYPE = getattr(hp, "encoder_type", "HyperNetwork")
    RENDERER_TYPE = getattr(hp, "renderer_type", "CNFRenderer")
    SAVE_PATH = getattr(hp, "save_path", "saved_models")
    USE_NODE_TYPE = False  # 强制不使用 node_type
    
    # Normalizer configs
    norm_cfg = getattr(hp, "normalizer", {})
    COORD_METHOD = norm_cfg.get("coord_method", "-11") if norm_cfg else "-11"
    FIELD_METHOD = norm_cfg.get("field_method", "ms") if norm_cfg else "ms"
    COORD_DIM = norm_cfg.get("coord_dim", None) if norm_cfg else None
    FIELD_DIM = norm_cfg.get("field_dim", None) if norm_cfg else None

    NORM_PARAMS_PATH = os.path.join(SAVE_PATH, "normalizer_params.pt")

    try:
        dataset = TrajectoryChunkDataset(
            dataset_path=DATASET_PATH,
            chunk_size=T_CHUNK,
            stride=STRIDE,
            use_vo=False,
            flatten=True,
            mode='test'
        )
        
        # 允许在配置文件中指定使用哪个 simulation (sim_idx)，如果不指定则默认随机选取一个
        import random
        # 尝试获取总模拟数，安全后退为 1
        num_sims = getattr(dataset, "num_sims", len(dataset) if hasattr(dataset, '__len__') else 1)
        target_sim_idx = getattr(hp, "sim_idx", random.randint(0, max(0, num_sims - 1)))
        print(f"Using dataset simulation index: {target_sim_idx}")
        if hasattr(dataset, "sim_indices"):
            dataset.sim_indices = [target_sim_idx]

        # coords shape is [N, 2]
        iter_dataset = iter(dataset)
        for i in range(7):
            original_coords_sample, traj_sample = next(iter_dataset)
        original_coords = original_coords_sample.unsqueeze(0).clone().detach().to(device) # expand to [1, N, 2]
        gt_fields_tensor = traj_sample.unsqueeze(0).clone().detach() # shape [1, T_CHUNK, N, C]
        node_type_tensor = None
    except Exception as e:
        print(f"Failed to load dataset coords: {e}")
        return
        
    # Setup Normalizers
    if os.path.exists(NORM_PARAMS_PATH):
        print(f"Loading normalizer parameters from {NORM_PARAMS_PATH}")
        params = torch.load(NORM_PARAMS_PATH, map_location=device, weights_only=False)
        coord_params = params['coord_params']
        field_params = params['field_params']
    else:
        print("Normalizer params not found. Cannot proceed without normalizers.")
        return
        
    coord_normalizer = Normalizer_ts(params=coord_params, method=COORD_METHOD, dim=COORD_DIM)
    field_normalizer = Normalizer_ts(params=field_params, method=FIELD_METHOD, dim=FIELD_DIM)

    # Init Models
    if ENCODER_TYPE == "HyperNetwork_Perceiver_v22":
        print("Using Perceiver_v22-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v22(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
            use_node_type=USE_NODE_TYPE
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_Perceiver_v23":
        print("Using Perceiver_v23-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v23(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
            use_node_type=USE_NODE_TYPE
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_GINO":
        print("Using GINO-based HyperNetwork")
        encoder = HyperNetwork_GINO(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            use_gino=getattr(hp, "use_gino", False),
            gno_radius=getattr(hp, "gno_radius", 0.05),
        ).to(device)
    else:
        print("Using standard HyperNetwork")
        encoder = HyperNetwork(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)
    
    if RENDERER_TYPE == "GaborRenderer_v22":
        print("Using Perceiver_v22-based GaborRenderer")
        cnf = GaborRenderer_v22(
            latent_dim=LATENT_DIM, 
            coord_dim=2, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF, 
            use_node_type=USE_NODE_TYPE
        ).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v23":
        print("Using Perceiver_v23-based GaborRenderer")
        cnf = GaborRenderer_v23(
            latent_dim=LATENT_DIM, 
            coord_dim=2, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF, 
            use_node_type=USE_NODE_TYPE
        ).to(device)
    elif RENDERER_TYPE == "GaborRenderer_GINO":
        print("Using GINO-based GaborRenderer")
        cnf = GaborRenderer_GINO(
            latent_dim=LATENT_DIM, 
            coord_dim=2, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF
        ).to(device)
    else:
        print("Using standard CNFRenderer")
        cnf = CNFRenderer(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)
    
    # Load trained weights from unified checkpoint format
    final_ckpt_path = os.path.join(SAVE_PATH, "checkpoint_final.pt")
    ckpt_path = None

    if os.path.exists(final_ckpt_path):
        ckpt_path = final_ckpt_path
    else:
        # Fallback to latest checkpoint_epoch_*.pt / checkpoint_step_*.pt
        ckpt_candidates = glob.glob(os.path.join(SAVE_PATH, "checkpoint_*.pt"))
        if ckpt_candidates:
            ckpt_path = max(ckpt_candidates, key=os.path.getmtime)

    if ckpt_path is not None:
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'encoder_ema_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_ema_state_dict'])
            print("Loaded encoder from encoder_ema_state_dict.")
        else:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
            print("Checkpoint has no EMA encoder, fallback to encoder_state_dict.")
            
        if 'cnf_ema_state_dict' in ckpt:
            cnf.load_state_dict(ckpt['cnf_ema_state_dict'])
            print("Loaded cnf from cnf_ema_state_dict.")
        else:
            cnf.load_state_dict(ckpt['cnf_state_dict'])
            print("Checkpoint has no EMA cnf, fallback to cnf_state_dict.")
            
        print(f"Checkpoint loaded (epoch={ckpt.get('epoch', 'N/A')}, global_step={ckpt.get('global_step', 'N/A')})")
    else:
        # Backward compatibility for legacy save format
        encoder_path = os.path.join(SAVE_PATH, "encoder.pth")
        cnf_path = os.path.join(SAVE_PATH, "cnf.pth")
        if os.path.exists(encoder_path) and os.path.exists(cnf_path):
            print("Loading legacy weights: encoder.pth and cnf.pth")
            encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
            cnf.load_state_dict(torch.load(cnf_path, map_location=device, weights_only=True))
        else:
            print(f"No checkpoint or legacy weights found in {SAVE_PATH}.")
            return
            
    encoder_params = sum(p.numel() for p in encoder.parameters())
    cnf_params = sum(p.numel() for p in cnf.parameters())
    print(f"Encoder model parameters: {encoder_params / 1e6:.2f}M ({encoder_params})")
    print(f"Renderer (CNF) model parameters: {cnf_params / 1e6:.2f}M ({cnf_params})")
    print(f"Total model parameters: {(encoder_params + cnf_params) / 1e6:.2f}M")
    
    encoder.eval()
    cnf.eval()
    
    with torch.no_grad():
        # 1. Generation begins from PURE NOISE in Data Space
        # We match N_points with the actual dataset coordinates
        B, T, N, C = 1, T_CHUNK, original_coords.shape[1], C_OUT
        seed = time() % 1000  # simple time-based seed for variability
        print(f"Using random seed: {seed:.0f} for noise generation")
        torch.manual_seed(seed)  # for reproducibility
        
        # Generation Mode (One-step vs Multi-step)
        SAMPLING_MODE = getattr(hp, "sampling_mode", "5-step")
        if SAMPLING_MODE == "one-step":
            num_steps = 1
            print("Mode: 1-step fast generation")
        else:
            num_steps = getattr(hp, "num_sampling_steps", 8)
            print(f"Mode: Multi-step consistency sampling ({num_steps} steps)")

        t_max = getattr(hp, "t_max", 80.0)
        t_min = getattr(hp, "t_min", 0.002)
        rho = 5.0
        
        # EDM target timestep schedule (Karras et al. 2022)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        t_steps = (t_max ** (1 / rho) + step_indices / (max(num_steps - 1, 1)) * (t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # Append t=0 for the final step condition
        
        x = torch.randn(B, T, N, C).to(device) * t_max
        coords = original_coords # [B, N, 2]
        coords_norm = coord_normalizer.normalize(coords)
        
        # Ground truth sequence normalized
        gt_fields_norm = field_normalizer.normalize(gt_fields_tensor.to(device))
        # Initial frame (t=0) to be used for guidance
        gt_init = gt_fields_norm[:, 0:1, :, :] # shape [B, 1, N, C]
        
        print(f"Starting {num_steps}-step consistency sampling loop with initial frame guidance...")
        
        # 使用基于梯度引导的后验采样(Diffusion Posterior Sampling)来确保时空连续性
        for i in range(num_steps):
            t_curr = t_steps[i]
            t_next = t_steps[i+1]
            
            t_target = torch.ones(B, device=device) * t_curr
            t_expand = t_target.view(B, 1, 1, 1)

            # EDM-style Skip Connection Scaling Factors
            sigma_data = 0.5
            c_in = 1.0 / torch.sqrt(t_expand**2 + sigma_data**2)
            
            # 引入梯度引导 (Gradient Guidance / DPS): 让网络自动调整无条件生成的潜在流形以使其符合条件
            x_in = x.detach().requires_grad_(True)
            
            with torch.enable_grad():
                # 4. Hyper-Network extracts the dynamic system latent
                if USE_NODE_TYPE:
                    z1_gen = encoder(c_in * x_in, coords_norm, t_target, node_type_tensor)
                    f_theta_guide = cnf(z1_gen, coords_norm, node_type_tensor)
                else:
                    z1_gen = encoder(c_in * x_in, coords_norm, t_target)
                    f_theta_guide = cnf(z1_gen, coords_norm)
                    
                # 计算第一帧与 GT 的差异 (MSE Loss)
                loss = torch.nn.functional.mse_loss(f_theta_guide[:, 0:1], gt_init)
                
                # 对当前噪声数据进行求导
                grad = torch.autograd.grad(loss, x_in)[0]
            
            # 自动调整自适应引导步长 (按当前噪声尺度缩放，基于无条件流形做微调)
            step_size = 0.5 * t_curr / (torch.norm(grad) + 1e-8)
            x_guided = x_in - step_size * grad
            
            # 用引导后的 x_guided 重新预测最终的 x0
            if USE_NODE_TYPE:
                z1_final = encoder(c_in * x_guided, coords_norm, t_target, node_type_tensor)
                x0_pred = cnf(z1_final, coords_norm, node_type_tensor)
            else:
                z1_final = encoder(c_in * x_guided, coords_norm, t_target)
                x0_pred = cnf(z1_final, coords_norm)
            
            
            if i < num_steps - 1:
                # 强化引导：将网络预测的去噪干净数据强制对齐真实初始帧，再添加下一步的联合去噪过程噪声
                x0_pred[:, 0:1, :, :] = gt_init
                # Add noise back to t_next
                noise = torch.randn_like(x_guided)
                # Following Consistency Models: x_{n-1} = x0_pred + sqrt(t_{prev}^2 - t_min^2) * z
                std = torch.sqrt(torch.clamp(t_next**2 - t_min**2, min=0.0))
                x = x0_pred + std * noise
            else:
                x = x0_pred

        # Ensure the final output perfectly matches the initial frame
        # x0_pred[:, 0:1, :, :] = gt_init
        
        trajectory_pred_norm = x0_pred
        
        # Denormalize output trajectory to physical space
        trajectory_pred = field_normalizer.denormalize(trajectory_pred_norm)
        
        print(f"Generated clean trajectory shape: {trajectory_pred.shape}")
        print("Success! One-step generation achieved via integrated CNF & Flow Data-space training.")
        
        # Save visualization directly after generation
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.tri as mtri
        import numpy as np

        field_pred = trajectory_pred.detach().cpu().numpy()  # [1, T_CHUNK, N, C]
        field_gt = gt_fields_tensor.cpu().numpy()            # [1, T_CHUNK, N, C]
        coord = coords[0].detach().cpu().numpy()
        
        frames = field_pred.shape[1]  # T_CHUNK
        
        x = coord[:, 0]
        y = coord[:, 1]
        
        tri = mtri.Triangulation(x, y)

        def get_vertex_values(data):
            if data.ndim == 2 and data.shape[1] > 1:
                return np.linalg.norm(data, axis=1)
            else:
                return np.abs(data.squeeze())

        values0_pred = get_vertex_values(field_pred[0, 0])
        values0_gt = get_vertex_values(field_gt[0, 0])
        
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 6))

        tpc_gt = ax_gt.tripcolor(tri, values0_gt, shading="gouraud", cmap="viridis")
        ax_gt.set_aspect('equal')
        ax_gt.set_xlabel("x")
        ax_gt.set_ylabel("y")
        title_gt = ax_gt.set_title("Ground Truth (time=0)")
        
        tpc_pred = ax_pred.tripcolor(tri, values0_pred, shading="gouraud", cmap="viridis")
        ax_pred.set_aspect('equal')
        ax_pred.set_xlabel("x")
        ax_pred.set_ylabel("y")
        title_pred = ax_pred.set_title("Prediction (time=0)")

        cbar = plt.colorbar(tpc_pred, ax=[ax_gt, ax_pred], fraction=0.03, pad=0.04, label="Velocity Magnitude |(u,v)|")

        def update(t):
            v_gt = get_vertex_values(field_gt[0, t])
            v_pred = get_vertex_values(field_pred[0, t])
            
            tpc_gt.set_array(v_gt)
            tpc_pred.set_array(v_pred)
            
            vmin = min(v_gt.min(), v_pred.min())
            vmax = max(v_gt.max(), v_pred.max())
            tpc_gt.set_clim(vmin, vmax)
            tpc_pred.set_clim(vmin, vmax)
            
            title_gt.set_text(f"Ground Truth (time={t})")
            title_pred.set_text(f"Prediction (time={t})")
            return tpc_gt, tpc_pred, title_gt, title_pred

        ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=80, blit=False)

        os.makedirs(SAVE_PATH, exist_ok=True)
        base_name = "generated_trajectory_guided"
        ext = ".gif"
        counter = 0
        gif_path = os.path.join(SAVE_PATH, f"{base_name}{ext}")
        while os.path.exists(gif_path):
            counter += 1
            gif_path = os.path.join(SAVE_PATH, f"{base_name}_{counter}{ext}")

        print(f"Saving generation animation to {gif_path} ...")
        ani.save(gif_path, writer="pillow", fps=12)
        print("Animation saved successfully.")

        # ==========================================================
        # 频域信息分析 (Frequency Domain Analysis)
        # ==========================================================
        print("Computing frequency domain energy spectra...")
        
        # 将张量转为 NumPy [T_CHUNK, N, C]
        field_np = trajectory_pred.detach().cpu().numpy()[0] 
        T_len = field_np.shape[0]
        num_channels = field_np.shape[2]
        
        fig_freq, ax_freq = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 时间轴频域能量 (Temporal Frequency Energy)
        # 对时间轴进行 1D FFT [T_CHUNK, N, C]
        fft_temporal = np.fft.fft(field_np, axis=0) 
        energy_temporal = np.abs(fft_temporal)**2
        
        # 计算平均能量并取出正频率部分
        freqs = np.fft.fftfreq(T_len)
        pos_freq_idxs = freqs > 0
        
        # 将频率对应为波数 (Wavenumber/Mode k)
        k_temporal = np.arange(1, np.sum(pos_freq_idxs) + 1)
        
        # 仅在 N 维度求平均，保留通道维度 C: 输出形状 [频率数, 通道数]
        energy_temporal_avg_N = np.mean(energy_temporal, axis=1)[pos_freq_idxs]
        
        for c in range(num_channels):
            ax_freq[0].plot(k_temporal, energy_temporal_avg_N[:, c], marker='o', linestyle='-', label=f"Channel {c}")
            
        ax_freq[0].set_yscale('log')
        # ax_freq[0].set_xscale('log')
        ax_freq[0].set_xlabel("Temporal Wavenumber ($k$)")
        ax_freq[0].set_ylabel("Average Energy")
        ax_freq[0].set_title("Temporal Energy Spectrum")
        ax_freq[0].grid(True, which="both", ls="--", alpha=0.5)
        ax_freq[0].legend()

        # 2. 空间轴频域能量 (Spatial Frequency Energy / Wavenumber Spectrum)
        # 取定初始时刻 (t=0) 的空间场通过插值为网格进行 2D FFT 估计物理场的能量级联
        try:
            from scipy.interpolate import griddata
            
            data0 = field_np[0] # [N, C]
            x_coords = coord[:, 0]
            y_coords = coord[:, 1]
            
            # 插值到 128x128 的规则网格
            grid_res = 128
            grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):complex(0, grid_res), min(y_coords):max(y_coords):complex(0, grid_res)]
            
            for c in range(num_channels):
                values_sp_c = data0[:, c]
                grid_z = griddata((x_coords, y_coords), values_sp_c, (grid_x, grid_y), method='linear')
                grid_z[np.isnan(grid_z)] = 0.0 # 填补 nan
                
                # 进行 2D FFT 并将其中心化
                fft2_spatial = np.fft.fft2(grid_z)
                fft2_spatial_shift = np.fft.fftshift(fft2_spatial)
                energy_spatial_2d = np.abs(fft2_spatial_shift)**2
                
                # 进行径向平均(Radial average)获得各波数(wavenumber)频段的一维能量谱
                center_x, center_y = grid_res//2, grid_res//2
                y_idx, x_idx = np.indices(grid_z.shape)
                r = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2).astype(int)
                
                tbin = np.bincount(r.ravel(), energy_spatial_2d.ravel())
                nr = np.bincount(r.ravel())
                radialprofile = tbin / np.maximum(nr, 1)
                
                k_vals = np.arange(1, len(radialprofile)) # 略去 k=0 的直流分量
                ax_freq[1].plot(k_vals, radialprofile[1:], marker='.', linestyle='-', label=f"Channel {c}")
                
            ax_freq[1].set_yscale('log')
            # ax_freq[1].set_xscale('log')
            ax_freq[1].set_xlabel("Wavenumber ($k$)")
            ax_freq[1].set_ylabel("Energy $E(k)$")
            ax_freq[1].set_title("Spatial Energy Spectrum (t=0)")
            ax_freq[1].grid(True, which="both", ls="--", alpha=0.5)
            ax_freq[1].legend()
        except Exception as e:
            print(f"Failed to compute spatial frequency: {e}")
            ax_freq[1].set_title("Spatial Spectral Failed")

        # 保存图像
        freq_save_path = os.path.join(SAVE_PATH, "frequency_energy_spectrum.png")
        plt.tight_layout()
        plt.savefig(freq_save_path, dpi=150)
        print(f"Saved frequency energy spectra to {freq_save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inference <config.yml>")
        sys.exit(1)
        
    hp = ri.basic_input(sys.argv[-1])
    inference_demo(hp)
