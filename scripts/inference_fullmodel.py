import sys
import os
import glob
import torch
from basicutility import ReadInput as ri
from src.dataset import TrajectoryChunkDataset
from src.models_v21 import FullModel_v21
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
    RENDERER_TYPE = getattr(hp, "renderer_type", "gabor")
    SAVE_PATH = getattr(hp, "save_path", "saved_models")
    
    # Normalizer configs
    norm_cfg = getattr(hp, "normalizer", {})
    COORD_METHOD = norm_cfg.get("coord_method", "-11") if norm_cfg else "-11"
    FIELD_METHOD = norm_cfg.get("field_method", "ms") if norm_cfg else "ms"
    COORD_DIM = norm_cfg.get("coord_dim", None) if norm_cfg else None
    FIELD_DIM = norm_cfg.get("field_dim", None) if norm_cfg else None

    NORM_PARAMS_PATH = os.path.join(SAVE_PATH, "normalizer_params.pt")

    try:
        from datasets import load_from_disk
        import numpy as np

        full_ds = load_from_disk(DATASET_PATH)
        total_len = len(full_ds)
        data_ratio = getattr(hp, "data_ratio", 1.0)
        
        sim_indices = list(range(total_len))
        if data_ratio < 1.0:
            np.random.seed(42)  # Set seed for reproducibility
            num_samples = max(1, int(total_len * data_ratio))
            sim_indices = np.random.choice(total_len, num_samples, replace=False).tolist()
            sim_indices.sort()
            
        dataset = TrajectoryChunkDataset(
            dataset_path=DATASET_PATH,
            chunk_size=T_CHUNK,
            use_vo=False,
            flatten=True,
            mode='test', # or train
            sim_indices=sim_indices
        )
        # coords shape is [N, 2]
        original_coords = dataset.coords.unsqueeze(0).to(device) # expand to [1, N, 2]
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
    print("Using FullModel_v21 Architecture")
    model = FullModel_v21(
        t_chunk=T_CHUNK,
        channel_in=C_OUT,
        channel_out=C_OUT,
        coord_dim=2,
        latent_dim=LATENT_DIM,
        time_emb_dim=256,
        hidden_dim=HIDDEN_DIM,
        num_heads=8,
        depth=DEPTH_ENC,
        num_tokens=NUM_TOKENS,
        fourier_dim=64,
        num_layers=NUM_LAYERS_CNF,
        renderer_type= RENDERER_TYPE,
        use_node_type=False
    ).to(device)
    
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
        if 'model_ema_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_ema_state_dict'])
            print("Loaded model from model_ema_state_dict.")
        elif 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print("Checkpoint has no EMA model, fallback to model_state_dict.")
        print(f"Checkpoint loaded (epoch={ckpt.get('epoch', 'N/A')}, global_step={ckpt.get('global_step', 'N/A')})")
    else:
        print(f"No checkpoint found in {SAVE_PATH}.")
        return
            
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {(model_params) / 1e6:.2f}M")
    
    model.eval()
    
    with torch.no_grad():
        # 1. Generation begins from PURE NOISE in Data Space
        # We match N_points with the actual dataset coordinates
        B, T, N, C = 1, T_CHUNK, original_coords.shape[1], C_OUT
        seed = time() % 10000  # simple time-based seed for variability
        print(f"Using random seed: {seed:.0f} for noise generation")
        torch.manual_seed(seed)  # for reproducibility
        x_noise = torch.randn(B, T, N, C).to(device)
        
        # 2. We want completely clean data, which corresponds to t=1.0 in our setup
        # The time step is scaled by 1000.0 for the model's SinusoidalPositionEmbeddings
        t_target = torch.ones(B).to(device)
        t_scaled = t_target * 1000.0
        
        # 3. We use original dataset coordinates for the noise support 
        coords = original_coords # [B, N, 2]
        
        # Normalize input coords
        coords_norm = coord_normalizer.normalize(coords)
        
        # 4. Neural Network extracts the dynamic system latent and renders the field
        # Note: the input noise is already standard normal, coords must be normalized
        trajectory_pred_norm = model(x_noise, t_scaled, input_coords=coords_norm, query_coords=coords_norm) # [1, T_CHUNK, N, C_OUT]
        
        # Denormalize output trajectory to physical space
        trajectory_pred = field_normalizer.denormalize(trajectory_pred_norm)
        
        print(f"Generated clean trajectory shape: {trajectory_pred.shape}")
        print("Success! One-step generation achieved via integrated FullModel Data-space training.")
        
        # Save visualization directly after generation
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.tri as mtri
        import numpy as np

        field = trajectory_pred.detach().cpu().numpy()
        coord = coords[0].detach().cpu().numpy()
        
        frames = field.shape[1]  # T_CHUNK
        data0 = field[0, 0]      # first batch, first timestep [N, C]
        
        if data0.ndim == 2 and data0.shape[1] > 1:
            values0 = np.linalg.norm(data0, axis=1)
        else:
            values0 = data0.squeeze()

        x = coord[:, 0]
        y = coord[:, 1]
        tri = mtri.Triangulation(x, y)

        fig, ax = plt.subplots(figsize=(10, 5))
        tpc = ax.tripcolor(tri, values0, shading="gouraud", cmap="viridis")
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        title = ax.set_title("Generated Field (time=0)")
        cbar = plt.colorbar(tpc, ax=ax, label="Velocity Magnitude")

        def update(t_val):
            data = field[0, t_val]
            if data.ndim == 2 and data.shape[1] > 1:
                values = np.linalg.norm(data, axis=1)
            else:
                values = data.squeeze()
            tpc.set_array(values)
            title.set_text(f"Generated Field (time={t_val})")
            return tpc, title

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
        print("Usage: python -m scripts.inference_fullmodel <config.yml>")
        sys.exit(1)
        
    hp = ri.basic_input(sys.argv[-1])
    inference_demo(hp)
