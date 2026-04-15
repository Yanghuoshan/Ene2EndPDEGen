import sys
import os
import glob
import torch
from basicutility import ReadInput as ri
from src.dataset import TrajectoryChunkDataset, H5DirectoryChunkDataset
from src.models import HyperNetwork, HyperNetwork_FA, HyperNetwork_AP, HyperNetwork_ST, HyperNetwork_Perceiver, CNFRenderer, GaborRenderer
from src.models_v2 import HyperNetwork_Perceiver_v2, GaborRenderer_v2, HyperNetwork_Perceiver_v3, GaborRenderer_v3, HyperNetwork_Perceiver_v4, GaborRenderer_v4, HyperNetwork_Perceiver_v5, GaborRenderer_v5
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
    USE_NODE_TYPE = getattr(hp, "use_node_type", False)
    
    # Normalizer configs
    norm_cfg = getattr(hp, "normalizer", {})
    COORD_METHOD = norm_cfg.get("coord_method", "-11") if norm_cfg else "-11"
    FIELD_METHOD = norm_cfg.get("field_method", "ms") if norm_cfg else "ms"
    COORD_DIM = norm_cfg.get("coord_dim", None) if norm_cfg else None
    FIELD_DIM = norm_cfg.get("field_dim", None) if norm_cfg else None

    NORM_PARAMS_PATH = os.path.join(SAVE_PATH, "normalizer_params.pt")

    try:
        dataset = H5DirectoryChunkDataset(
            dataset_path=DATASET_PATH,
            chunk_size=T_CHUNK,
            stride=STRIDE,
            mode='test', # or train
            return_mesh_info=True
        )
        
        # 允许在配置文件中指定使用哪个 simulation (sim_idx)，如果不指定则默认随机选取一个
        import random
        target_sim_idx = getattr(hp, "sim_idx", random.randint(0, dataset.num_sims - 1))
        print(f"Using dataset simulation index: {target_sim_idx}")
        dataset.sim_indices = [target_sim_idx]

        # coords shape is [N, 2]
        original_coords_sample, mesh_info = next(iter(dataset))
        original_coords = original_coords_sample.unsqueeze(0).clone().detach().to(device) # expand to [1, N, 2]
        cells_tensor = mesh_info['cells']
        gt_fields_tensor = mesh_info['fields'].unsqueeze(0).clone().detach() # shape [1, T_CHUNK, N, C]
        if USE_NODE_TYPE:
            node_type_tensor = mesh_info['node_type'].unsqueeze(0).clone().detach().to(device) # shape [1, N, 1]
        else:
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
    if ENCODER_TYPE == "HyperNetwork_FA":
        print("Using FA-based HyperNetwork")
        encoder = HyperNetwork_FA(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_AP":
        print("Using AP-based HyperNetwork")
        encoder = HyperNetwork_AP(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_ST":
        print("Using ST-based HyperNetwork")
        from src.models import HyperNetwork_ST
        encoder = HyperNetwork_ST(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_Perceiver":
        print("Using Perceiver-based HyperNetwork")
        from src.models import HyperNetwork_Perceiver
        encoder = HyperNetwork_Perceiver(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_Perceiver_v2":
        print("Using Perceiver_v2-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v2(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_Perceiver_v3":
        print("Using Perceiver_v3-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v3(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_Perceiver_v4":
        print("Using Perceiver_v4-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v4(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
            use_node_type=USE_NODE_TYPE,
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
    
    if RENDERER_TYPE == "GaborRenderer":
        print("Using MFN-based GaborRenderer")
        cnf = GaborRenderer(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v2":
        print("Using MFN_v2-based GaborRenderer")
        cnf = GaborRenderer_v2(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v3":
        print("Using MFN_v3-based GaborRenderer")
        cnf = GaborRenderer_v3(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v4":
        print("Using MFN_v4-based GaborRenderer")
        cnf = GaborRenderer_v4(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF, use_node_type=USE_NODE_TYPE).to(device)
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
        cnf.load_state_dict(ckpt['cnf_state_dict'])
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
    
    encoder.eval()
    cnf.eval()
    
    with torch.no_grad():
        # 1. 设置基础形状和坐标
        B, T, N, C = 1, T_CHUNK, original_coords.shape[1], C_OUT
        seed = int(time() % 10000)  # simple time-based seed for variability
        print(f"Using random seed: {seed} for noise generation")
        torch.manual_seed(seed)  # for reproducibility
        
        coords = original_coords # [B, N, 2]
        coords_norm = coord_normalizer.normalize(coords)
        
        # 2. 提取真实的初始帧(第一帧)作为引导条件
        x_init_clean = gt_fields_tensor[:, 0:1, :, :].to(device)
        x_init_clean_norm = field_normalizer.normalize(x_init_clean)
        
        # 3. 定义多步替换引导的时间调度 (从纯噪声 t=1.0 降至纯净 t=0.0)
        time_steps = [x/50.0 for x in range(50, -1, -1)]  # 可以根据需要调整步数和每步的噪声水平
        x_current = torch.randn(B, T, N, C).to(device)
        
        print(f"Starting multi-step guidance with steps: {time_steps}")
        for i in range(len(time_steps) - 1):
            t_curr = time_steps[i]    # 当前的噪声水平
            t_next = time_steps[i+1]  # 下一个较小的噪声水平
            
            # --- A. 给干净的第一帧加上当前水平的噪声 ---
            # 这里假定前向加噪过程为线性调度: x_t = (1-t)*x_0 + t*noise
            noise_for_init = torch.randn_like(x_init_clean_norm)
            x_init_t = (1.0 - t_curr) * x_init_clean_norm + t_curr * noise_for_init
            
            # --- B. 强制替换当前带噪序列的第一帧 (Replacement Trick) ---
            x_current[:, 0:1, :, :] = x_init_t
            
            # --- C. 预测出完全干净的全序列 ---
            t_tensor = (torch.ones(B) * t_curr).to(device)
            if USE_NODE_TYPE:
                z1_gen = encoder(x_current, coords_norm, t_tensor, node_type_tensor)
                x_0_pred = cnf(z1_gen, coords_norm, node_type_tensor)
            else:
                z1_gen = encoder(x_current, coords_norm, t_tensor)
                x_0_pred = cnf(z1_gen, coords_norm)
            
            # --- D. 退回倒下一步 (添加 t_next 水平的噪声) ---
            if t_next > 0.0:
                noise_resample = torch.randn_like(x_0_pred)
                x_current = (1.0 - t_next) * x_0_pred + t_next * noise_resample
            else:
                # 如果下一步是 0，预测的 x_0_pred 即可作为最终输出
                trajectory_pred_norm = x_0_pred
                
        # 4. 最后一步强行确保第一帧完美等于 ground truth
        trajectory_pred_norm[:, 0:1, :, :] = x_init_clean_norm
        
        # Denormalize output trajectory to physical space
        trajectory_pred = field_normalizer.denormalize(trajectory_pred_norm)
        
        print(f"Generated clean trajectory shape: {trajectory_pred.shape}")
        print("Success! Guided generation achieved using multi-step replacement trick.")
        
        # Save visualization directly after generation
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.tri as mtri
        import numpy as np

        field_pred = trajectory_pred.detach().cpu().numpy()  # [1, T_CHUNK, N, C]
        field_gt = gt_fields_tensor.cpu().numpy()            # [1, T_CHUNK, N, C]
        coord = coords[0].detach().cpu().numpy()
        cells = cells_tensor.detach().cpu().numpy()
        
        frames = field_pred.shape[1]  # T_CHUNK
        
        x = coord[:, 0]
        y = coord[:, 1]
        
        # cells dimensions: [T, N, 3] or [N, 3], we take the first frame's connectivity.
        if cells.ndim == 3:
            triangles = cells[0]
        else:
            triangles = cells

        tri = mtri.Triangulation(x, y, triangles)

        def get_face_values(data):
            # Compute velocity magnitude ||(u, v)|| from the first two channels.
            if data.ndim == 2 and data.shape[1] >= 2:
                values = np.linalg.norm(data[:, :2], axis=1)
            else:
                values = np.abs(data.squeeze())
            # Each triangle gets meant value of its 3 vertices
            return values[triangles].mean(axis=1)

        face_values0_pred = get_face_values(field_pred[0, 0])
        face_values0_gt = get_face_values(field_gt[0, 0])
        
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 6))

        # 统一和固定 colorbar 范围可以更好地比较，不过这里我们默认使用单独范围，或随时间自动更新。
        tpc_gt = ax_gt.tripcolor(tri, facecolors=face_values0_gt, cmap="viridis")
        ax_gt.triplot(tri, color='black', linewidth=0.2, alpha=0.5)
        ax_gt.set_aspect('equal')
        ax_gt.set_xlabel("x")
        ax_gt.set_ylabel("y")
        title_gt = ax_gt.set_title("Ground Truth (time=0)")
        
        tpc_pred = ax_pred.tripcolor(tri, facecolors=face_values0_pred, cmap="viridis")
        ax_pred.triplot(tri, color='black', linewidth=0.2, alpha=0.5)
        ax_pred.set_aspect('equal')
        ax_pred.set_xlabel("x")
        ax_pred.set_ylabel("y")
        title_pred = ax_pred.set_title("Prediction (time=0)")

        # 组合的 colorbar
        cbar = plt.colorbar(tpc_pred, ax=[ax_gt, ax_pred], fraction=0.03, pad=0.04, label="Velocity Magnitude |(u,v)|")

        def update(t):
            fv_gt = get_face_values(field_gt[0, t])
            fv_pred = get_face_values(field_pred[0, t])
            
            # 更新颜色
            tpc_gt.set_array(fv_gt)
            tpc_pred.set_array(fv_pred)
            
            # 动态更新 color limits 使其能够包容两者的最大最小值
            vmin = min(fv_gt.min(), fv_pred.min())
            vmax = max(fv_gt.max(), fv_pred.max())
            tpc_gt.set_clim(vmin, vmax)
            tpc_pred.set_clim(vmin, vmax)
            
            title_gt.set_text(f"Ground Truth (time={t})")
            title_pred.set_text(f"Prediction (time={t})")
            return tpc_gt, tpc_pred, title_gt, title_pred

        ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=80, blit=False)

        os.makedirs(SAVE_PATH, exist_ok=True)
        base_name = "generated_trajectory"
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
