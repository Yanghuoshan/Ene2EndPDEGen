import sys
import os
import glob
import torch
from basicutility import ReadInput as ri
from src.dataset import MHDChunkDataset
from src.models_v22 import HyperNetwork_Perceiver_v22, GaborRenderer_v22, HyperNetwork_Perceiver_v23, GaborRenderer_v23
from src.siren import SIRENRenderer
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

    try:
        dataset = MHDChunkDataset(
            dataset_path=DATASET_PATH,
            chunk_size=T_CHUNK,
            stride=getattr(hp, "stride", T_CHUNK),
            mode='test',
            downsample_factor=getattr(hp, "downsample_factor", 1) if getattr(hp, "use_downsample", False) else 1
        )
        coords_sample, gt_chunk = next(iter(dataset))
        # coords shape: [N, 3], gt_chunk shape: [T, N, C]
        original_coords = coords_sample.unsqueeze(0).to(device)
        gt_trajectory = gt_chunk.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Failed to load MHD dataset sample: {e}")
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
            coord_dim=3,
            use_node_type=False,
            use_flash_attn=getattr(hp, "use_flash_attn", True)
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
            coord_dim=3,
            use_node_type=False,
            use_flash_attn=getattr(hp, "use_flash_attn", True)
        ).to(device)
    else:
        raise ValueError(f"Unsupported ENCODER_TYPE for MHD inference: {ENCODER_TYPE}")
    
    if RENDERER_TYPE == "GaborRenderer_v22":
        print("Using GaborRenderer_v22")
        cnf = GaborRenderer_v22(
            latent_dim=LATENT_DIM, 
            coord_dim=3, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF, 
            use_node_type=False,
            use_flash_attn=getattr(hp, "use_flash_attn", True)
        ).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v23":
        print("Using GaborRenderer_v23")
        cnf = GaborRenderer_v23(
            latent_dim=LATENT_DIM, 
            coord_dim=3, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF, 
            use_node_type=False,
            use_flash_attn=getattr(hp, "use_flash_attn", True)
        ).to(device)
    elif RENDERER_TYPE == "SIREN":
        print("Using SIRENRenderer")
        cnf = SIRENRenderer(
            latent_dim=LATENT_DIM,
            coord_dim=3,
            t_chunk=T_CHUNK,
            channel_out=C_OUT,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS_CNF,
            use_node_type=False,
            use_flash_attn=getattr(hp, "use_flash_attn", True)
        ).to(device)
    else:
        raise ValueError(f"Unsupported RENDERER_TYPE for MHD inference: {RENDERER_TYPE}")
    
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
            num_steps = getattr(hp, "num_sampling_steps", 5)
            print(f"Mode: Multi-step consistency sampling ({num_steps} steps)")

        t_max = getattr(hp, "t_max", 80.0)
        t_min = getattr(hp, "t_min", 0.002)
        rho = 7.0
        
        # EDM target timestep schedule (Karras et al. 2022)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        t_steps = (t_max ** (1 / rho) + step_indices / (max(num_steps - 1, 1)) * (t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # Append t=0 for the final step condition
        
        x = torch.randn(B, T, N, C).to(device) * t_max
        coords = original_coords # [B, N, 3]
        coords_norm = coord_normalizer.normalize(coords)
        
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
                noise = torch.randn_like(x)
                # Following Consistency Models: x_{n-1} = x0_pred + sqrt(t_{prev}^2 - t_min^2) * z
                std = torch.sqrt(torch.clamp(t_next**2 - t_min**2, min=0.0))
                x = x0_pred + std * noise

        trajectory_pred_norm = x0_pred
        
        # Denormalize output trajectory to physical space
        trajectory_pred = field_normalizer.denormalize(trajectory_pred_norm)
        trajectory_gt = field_normalizer.denormalize(gt_trajectory)
        
        print(f"Generated clean trajectory shape: {trajectory_pred.shape}")
        print(f"Ground-truth trajectory shape: {trajectory_gt.shape}")
        print("Success! One-step generation achieved via integrated CNF & Flow Data-space training.")
        
        # Save visualization directly after generation
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
        from scipy.interpolate import griddata

        field_pred = trajectory_pred.detach().cpu().numpy()
        field_gt = trajectory_gt.detach().cpu().numpy()
        coord = coords[0].detach().cpu().numpy()
        
        frames = min(field_pred.shape[1], field_gt.shape[1])

        def to_vis_scalar(data_nc):
            # Visualize only velocity magnitude |v| for MHD channels [rho, Bx, By, Bz, Vx, Vy, Vz].
            if data_nc.ndim == 2 and data_nc.shape[1] >= 7:
                vel_mag = np.linalg.norm(data_nc[:, 4:7], axis=1)
                return vel_mag
            if data_nc.ndim == 2 and data_nc.shape[1] > 1:
                return np.linalg.norm(data_nc, axis=1)
            return data_nc.squeeze()

        # Compare several planes sliced along one axis (default: z-axis).
        axis_map = {"x": 0, "y": 1, "z": 2}
        slice_axis_name = str(getattr(hp, "vis_slice_axis", "z")).lower()
        if slice_axis_name not in axis_map:
            print(f"Invalid vis_slice_axis={slice_axis_name}, fallback to 'z'.")
            slice_axis_name = "z"
        slice_axis = axis_map[slice_axis_name]

        vis_num_planes = max(1, int(getattr(hp, "vis_num_planes", 3)))
        vis_plane_max_points = max(100, int(getattr(hp, "vis_plane_max_points", 8000)))

        rounded_axis_vals = np.round(coord[:, slice_axis], 8)
        unique_axis_vals = np.unique(rounded_axis_vals)
        if unique_axis_vals.size == 0:
            print("No valid coordinate values found for slicing.")
            return

        if unique_axis_vals.size <= vis_num_planes:
            selected_vals = unique_axis_vals
        else:
            selected_ids = np.linspace(0, unique_axis_vals.size - 1, vis_num_planes, dtype=int)
            selected_vals = unique_axis_vals[selected_ids]

        rng = np.random.default_rng(0)
        plane_indices = []
        plane_vals = []
        for val in selected_vals:
            idx = np.where(rounded_axis_vals == val)[0]
            if idx.size == 0:
                continue
            if idx.size > vis_plane_max_points:
                idx = np.sort(rng.choice(idx, size=vis_plane_max_points, replace=False))
            plane_indices.append(idx)
            plane_vals.append(float(val))

        if len(plane_indices) == 0:
            print("Failed to build slice planes from coordinates.")
            return

        keep_axes = [i for i in range(3) if i != slice_axis]
        axis_labels = ["x", "y", "z"]

        pred0_list = [to_vis_scalar(field_pred[0, 0][idx]) for idx in plane_indices]
        gt0_list = [to_vis_scalar(field_gt[0, 0][idx]) for idx in plane_indices]
        all0 = np.concatenate(pred0_list + gt0_list)
        vmin = float(np.min(all0))
        vmax = float(np.max(all0))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

        n_planes = len(plane_indices)
        fig, axes = plt.subplots(2, n_planes, figsize=(4.5 * n_planes, 8), squeeze=False)
        images = []
        plane_meshes = []
        vis_grid_res = max(32, int(getattr(hp, "vis_grid_res", 128)))

        for j, idx in enumerate(plane_indices):
            plane_coords = coord[idx]
            px = plane_coords[:, keep_axes[0]]
            py = plane_coords[:, keep_axes[1]]

            x_min, x_max = float(np.min(px)), float(np.max(px))
            y_min, y_max = float(np.min(py)), float(np.max(py))
            if abs(x_max - x_min) < 1e-12:
                x_max = x_min + 1e-12
            if abs(y_max - y_min) < 1e-12:
                y_max = y_min + 1e-12

            gx, gy = np.meshgrid(
                np.linspace(x_min, x_max, vis_grid_res),
                np.linspace(y_min, y_max, vis_grid_res),
                indexing='xy',
            )

            plane_points = np.stack([px, py], axis=1)

            pred_grid = griddata(plane_points, pred0_list[j], (gx, gy), method='linear')
            gt_grid = griddata(plane_points, gt0_list[j], (gx, gy), method='linear')

            if np.isnan(pred_grid).any():
                pred_grid_nn = griddata(plane_points, pred0_list[j], (gx, gy), method='nearest')
                pred_grid = np.where(np.isnan(pred_grid), pred_grid_nn, pred_grid)
            if np.isnan(gt_grid).any():
                gt_grid_nn = griddata(plane_points, gt0_list[j], (gx, gy), method='nearest')
                gt_grid = np.where(np.isnan(gt_grid), gt_grid_nn, gt_grid)

            im_pred = axes[0, j].imshow(
                pred_grid,
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
            )
            im_gt = axes[1, j].imshow(
                gt_grid,
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
            )
            images.append((im_pred, im_gt))
            plane_meshes.append((plane_points, gx, gy, x_min, x_max, y_min, y_max))

            axes[0, j].set_title(f"Pred {slice_axis_name}={plane_vals[j]:.3f}, t=0")
            axes[1, j].set_title(f"GT {slice_axis_name}={plane_vals[j]:.3f}, t=0")
            axes[1, j].set_xlabel(axis_labels[keep_axes[0]])
            axes[0, j].set_ylabel(axis_labels[keep_axes[1]])
            axes[1, j].set_ylabel(axis_labels[keep_axes[1]])

        fig.colorbar(images[0][0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="|v|")

        def update(t):
            updated = []
            for j, idx in enumerate(plane_indices):
                pred_t = to_vis_scalar(field_pred[0, t][idx])
                gt_t = to_vis_scalar(field_gt[0, t][idx])

                plane_points, gx, gy, _, _, _, _ = plane_meshes[j]
                pred_grid = griddata(plane_points, pred_t, (gx, gy), method='linear')
                gt_grid = griddata(plane_points, gt_t, (gx, gy), method='linear')
                if np.isnan(pred_grid).any():
                    pred_grid_nn = griddata(plane_points, pred_t, (gx, gy), method='nearest')
                    pred_grid = np.where(np.isnan(pred_grid), pred_grid_nn, pred_grid)
                if np.isnan(gt_grid).any():
                    gt_grid_nn = griddata(plane_points, gt_t, (gx, gy), method='nearest')
                    gt_grid = np.where(np.isnan(gt_grid), gt_grid_nn, gt_grid)

                im_pred, im_gt = images[j]
                im_pred.set_data(pred_grid)
                im_gt.set_data(gt_grid)
                axes[0, j].set_title(f"Pred {slice_axis_name}={plane_vals[j]:.3f}, t={t}")
                axes[1, j].set_title(f"GT {slice_axis_name}={plane_vals[j]:.3f}, t={t}")
                updated.extend([im_pred, im_gt])
            return updated

        ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=80, blit=False)

        os.makedirs(SAVE_PATH, exist_ok=True)
        base_name = "generated_vs_gt_3d"
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
        # 频域信息分析 (Frequency Domain Analysis): Pred vs GT
        # ==========================================================
        print("Computing temporal frequency spectra (Prediction vs GT)...")

        pred_np = field_pred[0] # [T, N, C]
        gt_np = field_gt[0]     # [T, N, C]
        T_len = min(pred_np.shape[0], gt_np.shape[0])
        pred_np = pred_np[:T_len]
        gt_np = gt_np[:T_len]
        num_channels = pred_np.shape[2]

        fig_freq, ax_freq = plt.subplots(1, 2, figsize=(13, 5))

        freqs = np.fft.fftfreq(T_len)
        pos_freq_idxs = freqs > 0
        k_temporal = np.arange(1, np.sum(pos_freq_idxs) + 1)

        fft_temporal_pred = np.fft.fft(pred_np, axis=0)
        fft_temporal_gt = np.fft.fft(gt_np, axis=0)
        energy_temporal_pred = np.abs(fft_temporal_pred) ** 2
        energy_temporal_gt = np.abs(fft_temporal_gt) ** 2

        energy_temporal_pred_avg = np.mean(energy_temporal_pred, axis=1)[pos_freq_idxs]
        energy_temporal_gt_avg = np.mean(energy_temporal_gt, axis=1)[pos_freq_idxs]

        max_channels_to_plot = min(num_channels, 4)
        for c in range(max_channels_to_plot):
            ax_freq[0].plot(k_temporal, energy_temporal_pred_avg[:, c], linestyle='-', label=f"Pred C{c}")
            ax_freq[0].plot(k_temporal, energy_temporal_gt_avg[:, c], linestyle='--', label=f"GT C{c}")

        ax_freq[0].set_yscale('log')
        ax_freq[0].set_xlabel("Temporal Wavenumber ($k$)")
        ax_freq[0].set_ylabel("Average Energy")
        ax_freq[0].set_title("Temporal Energy Spectrum (Pred vs GT)")
        ax_freq[0].grid(True, which="both", ls="--", alpha=0.5)
        ax_freq[0].legend()

        # Time-wise scalar statistics comparison based on |v|
        pred_scalar_t = np.mean(to_vis_scalar(pred_np.reshape(-1, num_channels)).reshape(T_len, -1), axis=1)
        gt_scalar_t = np.mean(to_vis_scalar(gt_np.reshape(-1, num_channels)).reshape(T_len, -1), axis=1)
        rel_err_t = np.abs(pred_scalar_t - gt_scalar_t) / (np.abs(gt_scalar_t) + 1e-8)

        ax_freq[1].plot(np.arange(T_len), pred_scalar_t, label="Pred mean |v|", linestyle='-')
        ax_freq[1].plot(np.arange(T_len), gt_scalar_t, label="GT mean |v|", linestyle='--')
        ax_freq[1].plot(np.arange(T_len), rel_err_t, label="Relative error", linestyle='-.')
        ax_freq[1].set_xlabel("Time index")
        ax_freq[1].set_title("|v| Statistic Over Time")
        ax_freq[1].grid(True, ls="--", alpha=0.5)
        ax_freq[1].legend()

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
