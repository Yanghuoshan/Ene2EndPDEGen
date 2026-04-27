import sys
import os
import glob
import torch
from basicutility import ReadInput as ri
from src.dataset import IgnitHitDataset
from src.models import HyperNetwork, CNFRenderer
from src.siren import SIRENRenderer
from src.models_v22 import HyperNetwork_Perceiver_v22, GaborRenderer_v22, HyperNetwork_Perceiver_v23, GaborRenderer_v23
from src.models_ae import HyperNetwork_GINO, GaborRenderer_GINO
from src.normalize import Normalizer_ts
from src.utils import generate_spatial_grf
from time import time

def inference_demo(hp):
    """
    Demonstrates how generation works after the model is fully trained.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 0. Configuration Settings
    DATASET_PATH = hp.dataset_path
    
    T_CHUNK = getattr(hp, "chunk_size", 16)
    C_OUT = getattr(hp, "c_out", 12)
    LATENT_DIM = getattr(hp, "latent_dim", 256)
    HIDDEN_DIM = getattr(hp, "hidden_dim", 256)
    DEPTH_ENC = getattr(hp, "depth_enc", 4)
    NUM_TOKENS = getattr(hp, "num_tokens", 16)
    NUM_LAYERS_CNF = getattr(hp, "num_layers_cnf", 4)
    ENCODER_TYPE = getattr(hp, "encoder_type", "HyperNetwork")
    RENDERER_TYPE = getattr(hp, "renderer_type", "CNFRenderer")
    SAVE_PATH = getattr(hp, "save_path", "saved_models")
    
    # Normalizer configs
    norm_cfg = getattr(hp, "normalizer", {})
    COORD_METHOD = norm_cfg.get("coord_method", "-11") if norm_cfg else "-11"
    FIELD_METHOD = norm_cfg.get("field_method", "ms") if norm_cfg else "ms"
    COORD_DIM = norm_cfg.get("coord_dim", None) if norm_cfg else None
    FIELD_DIM = norm_cfg.get("field_dim", None) if norm_cfg else None

    NORM_PARAMS_PATH = os.path.join(SAVE_PATH, "normalizer_params.pt")
    GT_TRAJECTORY_IDX = int(getattr(hp, "gt_trajectory_idx", getattr(hp, "gt_sample_idx", 0)))

    try:
        STRIDE = getattr(hp, "stride", T_CHUNK)
        dataset = IgnitHitDataset(
            dataset_path=DATASET_PATH,
            chunk_size=T_CHUNK,
            stride=STRIDE,
            mode='test' # or train
        )
        # coords shape is [N, 2]
        original_coords = dataset.coords.unsqueeze(0).to(device) # expand to [1, N, 2]
    except Exception as e:
        print(f"Failed to load dataset coords: {e}")
        return
        
    # Load ground truth from the dataset
    print(f"Loading ground truth trajectory index {GT_TRAJECTORY_IDX} from test dataset...")
    ground_truth_sample = None
    try:
        dataset_iter = iter(dataset)
        coords_gt = None
        fields_gt = None
        for _ in range(GT_TRAJECTORY_IDX + 1):
            coords_gt, fields_gt = next(dataset_iter)
        # coords_gt: [N, 2], fields_gt: [T_CHUNK, N, C_OUT]
        ground_truth_sample = fields_gt.unsqueeze(0).to(device)  # [1, T_CHUNK, N, C_OUT]
        print(f"Ground truth trajectory loaded: shape {ground_truth_sample.shape}")
    except StopIteration:
        print(f"Warning: trajectory index {GT_TRAJECTORY_IDX} is out of range for the dataset")
        ground_truth_sample = None
        
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
            use_node_type=getattr(hp, "use_node_type", False)
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
            use_node_type=getattr(hp, "use_node_type", False)
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
            use_node_type=getattr(hp, "use_node_type", False)
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
            use_node_type=getattr(hp, "use_node_type", False)
        ).to(device)
    elif RENDERER_TYPE == "SIREN":
        print("Using SIREN Renderer")
        cnf = SIRENRenderer(
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
        seed = int(time() % 1000)  # simple time-based seed for variability
        print(f"Using random seed: {seed} for noise generation")
        torch.manual_seed(seed)  # for reproducibility
        
        # Generation Mode (One-step vs Multi-step)
        SAMPLING_MODE = getattr(hp, "sampling_mode", "5-step")
        if SAMPLING_MODE == "one-step":
            num_steps = 1
            print("Mode: 1-step fast generation")
        else:
            num_steps = getattr(hp, "num_sampling_steps", 10)
            print(f"Mode: Multi-step consistency sampling ({num_steps} steps)")

        t_max = getattr(hp, "t_max", 80.0)
        t_min = getattr(hp, "t_min", 0.002)
        rho = 7.0
        
        # EDM target timestep schedule (Karras et al. 2022)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        t_steps = (t_max ** (1 / rho) + step_indices / (max(num_steps - 1, 1)) * (t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # Append t=0 for the final step condition
        
        coords = original_coords # [B, N, 2]
        coords_norm = coord_normalizer.normalize(coords)
        x = generate_spatial_grf(coords_norm, target_shape=(B, T, N, C), length_scale=0.15, grid_size=64).to(device) * t_max
        
        # --- Unconditional Generation ---
        gt_init = None
        
        print(f"Starting {num_steps}-step consistency sampling loop...")
        for i in range(num_steps):
            t_curr = t_steps[i]
            t_next = t_steps[i+1]
            
            t_target = torch.ones(B, device=device) * t_curr
            t_expand = t_target.view(B, 1, 1, 1)

            # EDM-style Skip Connection Scaling Factors
            sigma_data = 0.5
            c_in = 1.0 / torch.sqrt(t_expand**2 + sigma_data**2)
            
            # 4. Hyper-Network extracts the dynamic system latent
            z1_gen = encoder(c_in * x, coords_norm, t_target)
            
            # 5. Render fields
            f_theta = cnf(z1_gen, coords_norm) # [1, T_CHUNK, N, C_OUT]
            
            # Direct prediction (no skip connection)
            x0_pred = f_theta
            
            if i < num_steps - 1:
                # Add noise back to t_next
                noise = generate_spatial_grf(coords_norm, target_shape=x.shape, length_scale=0.15, grid_size=64).to(device)
                # Following Consistency Models: x_{n-1} = x0_pred + sqrt(t_{prev}^2 - t_min^2) * z
                std = torch.sqrt(torch.clamp(t_next**2 - t_min**2, min=0.0))
                x = x0_pred + std * noise

        trajectory_pred_norm = x0_pred
        
        # Denormalize output trajectory to physical space
        trajectory_pred = field_normalizer.denormalize(trajectory_pred_norm)
        
        print(f"Generated clean trajectory shape: {trajectory_pred.shape}")
        print("Success! One-step generation achieved via integrated CNF & Flow Data-space training.")
        
        # Save visualization directly after generation
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        import numpy as np

        field = trajectory_pred.detach().cpu().numpy()
        coord = coords[0].detach().cpu().numpy()
        
        frames = field.shape[1]  # T_CHUNK

        # Visualization channel selection (for IgnitHit 12 channels)
        VIS_CHANNEL = int(getattr(hp, "vis_channel", getattr(hp, "visualize_channel", 0)))
        if VIS_CHANNEL < 0 or VIS_CHANNEL >= field.shape[-1]:
            print(f"vis_channel={VIS_CHANNEL} is out of range [0, {field.shape[-1]-1}], fallback to 0")
            VIS_CHANNEL = 0
        channel_names = getattr(hp, "channel_names", None)
        if isinstance(channel_names, (list, tuple)) and VIS_CHANNEL < len(channel_names):
            vis_channel_name = str(channel_names[VIS_CHANNEL])
        else:
            vis_channel_name = f"Channel {VIS_CHANNEL}"
        print(f"Visualization channel selected: {vis_channel_name} (index={VIS_CHANNEL})")

        # Structured grid dimensions inference
        x_uniq = np.unique(coord[:, 0])
        y_uniq = np.unique(coord[:, 1])
        N_x = len(x_uniq)
        N_y = len(y_uniq)
        x_min, x_max = float(np.min(coord[:, 0])), float(np.max(coord[:, 0]))
        y_min, y_max = float(np.min(coord[:, 1])), float(np.max(coord[:, 1]))
        
        # Prepare ground truth field if available
        field_gt = None
        if ground_truth_sample is not None:
            field_gt = ground_truth_sample.detach().cpu().numpy()
        
        # Determine color limits based on both GT and prediction
        all_values_pred = field[0, ..., VIS_CHANNEL]
        
        if field_gt is not None:
            all_values_gt = field_gt[0, ..., VIS_CHANNEL]
            # Use combined min/max for consistent color scale
            vmin = min(np.min(all_values_pred), np.min(all_values_gt))
            vmax = max(np.max(all_values_pred), np.max(all_values_gt))
        else:
            vmin = np.min(all_values_pred)
            vmax = np.max(all_values_pred)
        
        # Create comparison visualization
        if field_gt is not None:
            # Side-by-side comparison: GT | Prediction | Error
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            data_gt_0 = field_gt[0, 0]
            values_gt_0 = data_gt_0[:, VIS_CHANNEL]
            values_gt_0_2d = values_gt_0.reshape(N_x, N_y)

            data_pred_0 = field[0, 0]
            values_pred_0 = data_pred_0[:, VIS_CHANNEL]
            values_pred_0_2d = values_pred_0.reshape(N_x, N_y)

            error_0_2d = np.abs(values_pred_0_2d - values_gt_0_2d)

            im_gt = axes[0].imshow(values_gt_0_2d.T, cmap='RdBu_r', origin='lower', aspect='auto',
                                   extent=[x_min, x_max, y_min, y_max], vmin=vmin, vmax=vmax)
            axes[0].set_title(f'Ground Truth at t=0 ({vis_channel_name})')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            fig.colorbar(im_gt, ax=axes[0], label=vis_channel_name)

            im_pred = axes[1].imshow(values_pred_0_2d.T, cmap='RdBu_r', origin='lower', aspect='auto',
                                     extent=[x_min, x_max, y_min, y_max], vmin=vmin, vmax=vmax)
            axes[1].set_title(f'Generated at t=0 ({vis_channel_name})')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            fig.colorbar(im_pred, ax=axes[1], label=vis_channel_name)

            im_error = axes[2].imshow(error_0_2d.T, cmap='hot', origin='lower', aspect='auto',
                                      extent=[x_min, x_max, y_min, y_max])
            axes[2].set_title('Absolute Error at t=0')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('y')
            fig.colorbar(im_error, ax=axes[2], label='|Error|')
            
            def update_comparison(t):
                # Ground Truth
                data_gt = field_gt[0, t]
                values_gt = data_gt[:, VIS_CHANNEL]
                values_gt_2d = values_gt.reshape(N_x, N_y)

                # Prediction
                data_pred = field[0, t]
                values_pred = data_pred[:, VIS_CHANNEL]
                values_pred_2d = values_pred.reshape(N_x, N_y)

                error_2d = np.abs(values_pred_2d - values_gt_2d)

                im_gt.set_data(values_gt_2d.T)
                axes[0].set_title(f'Ground Truth at t={t} ({vis_channel_name})')

                im_pred.set_data(values_pred_2d.T)
                axes[1].set_title(f'Generated at t={t} ({vis_channel_name})')

                im_error.set_data(error_2d.T)
                axes[2].set_title(f'Absolute Error at t={t}')
                
                return im_gt, im_pred, im_error
            
            print("Generating comparison animation with GT, Prediction, and Error...")
            ani = animation.FuncAnimation(fig, update_comparison, frames=range(frames), interval=80, blit=False)
            
            os.makedirs(SAVE_PATH, exist_ok=True)
            base_name = "comparison_trajectory"
            ext = ".gif"
            counter = 0
            gif_path = os.path.join(SAVE_PATH, f"{base_name}{ext}")
            while os.path.exists(gif_path):
                counter += 1
                gif_path = os.path.join(SAVE_PATH, f"{base_name}_{counter}{ext}")
            
            print(f"Saving comparison animation to {gif_path} ...")
            ani.save(gif_path, writer="pillow", fps=12)
            print("Comparison animation saved successfully.")
        else:
            # Original single visualization if no GT available
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Initial plot for colorbar
            data = field[0, 0]
            values = data[:, VIS_CHANNEL]
            values_2d = values.reshape(N_x, N_y)
            
            im = ax.imshow(values_2d.T, cmap='RdBu_r', origin='lower', aspect='auto', 
                           extent=[x_min, x_max, y_min, y_max], vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, label=vis_channel_name)
            
            def update(t):
                data = field[0, t]
                values = data[:, VIS_CHANNEL]
                values_2d = values.reshape(N_x, N_y)
                im.set_data(values_2d.T)
                ax.set_title(f'Generated Field ({vis_channel_name}) at time step {t}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                return im,
            
            print("Generating animation with 2D heatplot projection...")
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
        
        # Comparison with ground truth spectra if available
        if field_gt is not None:
            field_gt_np = field_gt[0]
            fig_freq, ax_freq = plt.subplots(2, 2, figsize=(14, 10))
            plot_gt_spectra = True
        else:
            fig_freq, ax_freq = plt.subplots(1, 2, figsize=(12, 5))
            ax_freq = ax_freq.reshape(1, 2)
            plot_gt_spectra = False
        
        # 1. 时间轴频域能量 (Temporal Frequency Energy)
        # 对时间轴进行 1D FFT [T_CHUNK, N, C]
        fft_temporal = np.fft.fft(field_np, axis=0) 
        energy_temporal = np.abs(fft_temporal)**2
        
        # Ground truth temporal FFT if available
        if plot_gt_spectra:
            fft_temporal_gt = np.fft.fft(field_gt_np, axis=0)
            energy_temporal_gt = np.abs(fft_temporal_gt)**2
        
        # 计算平均能量并取出正频率部分
        freqs = np.fft.fftfreq(T_len)
        pos_freq_idxs = freqs > 0
        
        # 将频率对应为波数 (Wavenumber/Mode k)
        k_temporal = np.arange(1, np.sum(pos_freq_idxs) + 1)
        
        # 仅在 N 维度求平均，保留通道维度 C: 输出形状 [频率数, 通道数]
        energy_temporal_avg_N = np.mean(energy_temporal, axis=1)[pos_freq_idxs]
        
        for c in range(num_channels):
            ax_freq[0, 0].plot(k_temporal, energy_temporal_avg_N[:, c], marker='o', linestyle='-', label=f"Generated Ch {c}")
        
        if plot_gt_spectra:
            energy_temporal_avg_N_gt = np.mean(energy_temporal_gt, axis=1)[pos_freq_idxs]
            for c in range(num_channels):
                ax_freq[0, 0].plot(k_temporal, energy_temporal_avg_N_gt[:, c], marker='s', linestyle='--', 
                                  label=f"GT Ch {c}", alpha=0.7)
            
        ax_freq[0, 0].set_yscale('log')
        ax_freq[0, 0].set_xlabel("Temporal Wavenumber ($k$)")
        ax_freq[0, 0].set_ylabel("Average Energy")
        ax_freq[0, 0].set_title("Temporal Energy Spectrum")
        ax_freq[0, 0].grid(True, which="both", ls="--", alpha=0.5)
        ax_freq[0, 0].legend()

        # Plot temporal error spectrum if GT available
        if plot_gt_spectra:
            for c in range(num_channels):
                error_temporal = np.abs(energy_temporal_avg_N[:, c] - energy_temporal_avg_N_gt[:, c])
                ax_freq[0, 1].plot(k_temporal, error_temporal, marker='o', linestyle='-', label=f"Channel {c}")
            ax_freq[0, 1].set_yscale('log')
            ax_freq[0, 1].set_xlabel("Temporal Wavenumber ($k$)")
            ax_freq[0, 1].set_ylabel("Energy Difference")
            ax_freq[0, 1].set_title("Temporal Energy Error Spectrum")
            ax_freq[0, 1].grid(True, which="both", ls="--", alpha=0.5)
            ax_freq[0, 1].legend()

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
                spatial_ax_row = 1 if not plot_gt_spectra else 1
                spatial_ax_col = 1 if not plot_gt_spectra else 0
                ax_freq[spatial_ax_row, spatial_ax_col].plot(k_vals, radialprofile[1:], marker='.', linestyle='-', label=f"Generated Ch {c}")
            
            # Plot ground truth spatial spectrum if available
            if plot_gt_spectra:
                data0_gt = field_gt_np[0]
                for c in range(num_channels):
                    values_sp_c_gt = data0_gt[:, c]
                    grid_z_gt = griddata((x_coords, y_coords), values_sp_c_gt, (grid_x, grid_y), method='linear')
                    grid_z_gt[np.isnan(grid_z_gt)] = 0.0
                    
                    fft2_spatial_gt = np.fft.fft2(grid_z_gt)
                    fft2_spatial_shift_gt = np.fft.fftshift(fft2_spatial_gt)
                    energy_spatial_2d_gt = np.abs(fft2_spatial_shift_gt)**2
                    
                    center_x, center_y = grid_res//2, grid_res//2
                    y_idx, x_idx = np.indices(grid_z_gt.shape)
                    r = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2).astype(int)
                    
                    tbin_gt = np.bincount(r.ravel(), energy_spatial_2d_gt.ravel())
                    nr_gt = np.bincount(r.ravel())
                    radialprofile_gt = tbin_gt / np.maximum(nr_gt, 1)
                    
                    k_vals = np.arange(1, len(radialprofile_gt))
                    ax_freq[1, 0].plot(k_vals, radialprofile_gt[1:], marker='s', linestyle='--', label=f"GT Ch {c}", alpha=0.7)
                
                # Plot spatial error spectrum
                for c in range(num_channels):
                    values_sp_c = data0[:, c]
                    values_sp_c_gt = data0_gt[:, c]
                    grid_z = griddata((x_coords, y_coords), values_sp_c, (grid_x, grid_y), method='linear')
                    grid_z_gt = griddata((x_coords, y_coords), values_sp_c_gt, (grid_x, grid_y), method='linear')
                    grid_z[np.isnan(grid_z)] = 0.0
                    grid_z_gt[np.isnan(grid_z_gt)] = 0.0
                    
                    fft2_spatial = np.fft.fft2(grid_z)
                    fft2_spatial_shift = np.fft.fftshift(fft2_spatial)
                    energy_spatial_2d = np.abs(fft2_spatial_shift)**2
                    
                    fft2_spatial_gt = np.fft.fft2(grid_z_gt)
                    fft2_spatial_shift_gt = np.fft.fftshift(fft2_spatial_gt)
                    energy_spatial_2d_gt = np.abs(fft2_spatial_shift_gt)**2
                    
                    center_x, center_y = grid_res//2, grid_res//2
                    y_idx, x_idx = np.indices(grid_z.shape)
                    r = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2).astype(int)
                    
                    tbin = np.bincount(r.ravel(), energy_spatial_2d.ravel())
                    nr = np.bincount(r.ravel())
                    radialprofile = tbin / np.maximum(nr, 1)
                    
                    tbin_gt = np.bincount(r.ravel(), energy_spatial_2d_gt.ravel())
                    radialprofile_gt = tbin_gt / np.maximum(nr, 1)
                    
                    error_spatial = np.abs(radialprofile - radialprofile_gt)
                    k_vals = np.arange(1, len(error_spatial))
                    ax_freq[1, 1].plot(k_vals, error_spatial[1:], marker='.', linestyle='-', label=f"Channel {c}")
                
                ax_freq[1, 1].set_yscale('log')
                ax_freq[1, 1].set_xlabel("Wavenumber ($k$)")
                ax_freq[1, 1].set_ylabel("Energy Difference $|E_{pred}(k) - E_{GT}(k)|$")
                ax_freq[1, 1].set_title("Spatial Energy Error Spectrum (t=0)")
                ax_freq[1, 1].grid(True, which="both", ls="--", alpha=0.5)
                ax_freq[1, 1].legend()
                
            spatial_ax_row = 1 if not plot_gt_spectra else 1
            spatial_ax_col = 1 if not plot_gt_spectra else 0
            ax_freq[spatial_ax_row, spatial_ax_col].set_yscale('log')
            ax_freq[spatial_ax_row, spatial_ax_col].set_xlabel("Wavenumber ($k$)")
            ax_freq[spatial_ax_row, spatial_ax_col].set_ylabel("Energy $E(k)$")
            ax_freq[spatial_ax_row, spatial_ax_col].set_title("Spatial Energy Spectrum - Generated (t=0)")
            ax_freq[spatial_ax_row, spatial_ax_col].grid(True, which="both", ls="--", alpha=0.5)
            ax_freq[spatial_ax_row, spatial_ax_col].legend()
        except Exception as e:
            print(f"Failed to compute spatial frequency: {e}")
            spatial_ax_row = 1 if not plot_gt_spectra else 1
            spatial_ax_col = 1 if not plot_gt_spectra else 0
            ax_freq[spatial_ax_row, spatial_ax_col].set_title("Spatial Spectral Failed")

        # 保存图像
        freq_save_path = os.path.join(SAVE_PATH, "frequency_energy_spectrum.png")
        plt.tight_layout()
        plt.savefig(freq_save_path, dpi=150)
        print(f"Saved frequency energy spectra to {freq_save_path}")

        # ==========================================================
        # 误差指标计算 (Error Metrics Computation)
        # ==========================================================
        if ground_truth_sample is not None:
            print("\n" + "="*60)
            print("ERROR METRICS COMPARISON")
            print("="*60)
            
            trajectory_pred_np = trajectory_pred.detach().cpu().numpy()
            trajectory_gt_np = ground_truth_sample.detach().cpu().numpy()
            
            # Compute errors over all timesteps
            error_l2 = np.sqrt(np.mean((trajectory_pred_np - trajectory_gt_np)**2))
            error_l1 = np.mean(np.abs(trajectory_pred_np - trajectory_gt_np))
            error_linf = np.max(np.abs(trajectory_pred_np - trajectory_gt_np))
            
            # Per-channel errors
            error_per_channel = []
            for c in range(C_OUT):
                mse_c = np.mean((trajectory_pred_np[..., c] - trajectory_gt_np[..., c])**2)
                mae_c = np.mean(np.abs(trajectory_pred_np[..., c] - trajectory_gt_np[..., c]))
                error_per_channel.append({'mse': mse_c, 'mae': mae_c, 'rmse': np.sqrt(mse_c)})
            
            print(f"\nGlobal Error Metrics:")
            print(f"  MSE (L2^2):  {error_l2**2:.6e}")
            print(f"  RMSE (L2):   {error_l2:.6e}")
            print(f"  MAE (L1):    {error_l1:.6e}")
            print(f"  Max Error:   {error_linf:.6e}")
            
            for c, err in enumerate(error_per_channel):
                channel_names = ['Vorticity', 'Height']
                ch_name = channel_names[c] if c < len(channel_names) else f"Channel {c}"
                print(f"\n{ch_name}:")
                print(f"  RMSE:  {err['rmse']:.6e}")
                print(f"  MAE:   {err['mae']:.6e}")
                print(f"  MSE:   {err['mse']:.6e}")
            
            # Temporal error evolution
            print(f"\nTemporal Error Evolution:")
            for t in range(min(3, frames)):  # Show first 3 timesteps
                mse_t = np.mean((trajectory_pred_np[0, t] - trajectory_gt_np[0, t])**2)
                print(f"  t={t}: RMSE = {np.sqrt(mse_t):.6e}")
            
            # Save metrics to file
            metrics_path = os.path.join(SAVE_PATH, "error_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write("ERROR METRICS COMPARISON\n")
                f.write("="*60 + "\n\n")
                f.write(f"Global Error Metrics:\n")
                f.write(f"  MSE (L2^2):  {error_l2**2:.6e}\n")
                f.write(f"  RMSE (L2):   {error_l2:.6e}\n")
                f.write(f"  MAE (L1):    {error_l1:.6e}\n")
                f.write(f"  Max Error:   {error_linf:.6e}\n\n")
                
                for c, err in enumerate(error_per_channel):
                    channel_names = ['Vorticity', 'Height']
                    ch_name = channel_names[c] if c < len(channel_names) else f"Channel {c}"
                    f.write(f"{ch_name}:\n")
                    f.write(f"  RMSE:  {err['rmse']:.6e}\n")
                    f.write(f"  MAE:   {err['mae']:.6e}\n")
                    f.write(f"  MSE:   {err['mse']:.6e}\n\n")
                
                f.write(f"Temporal Error Evolution:\n")
                for t in range(frames):
                    mse_t = np.mean((trajectory_pred_np[0, t] - trajectory_gt_np[0, t])**2)
                    f.write(f"  t={t}: RMSE = {np.sqrt(mse_t):.6e}\n")
            
            print(f"\nMetrics saved to {metrics_path}")
            print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inference <config.yml>")
        sys.exit(1)
        
    hp = ri.basic_input(sys.argv[-1])
    inference_demo(hp)
