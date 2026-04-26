import sys
import os
import glob
import torch
from basicutility import ReadInput as ri
from src.dataset import TrajectoryChunkDataset, H5DirectoryChunkDataset
from src.models import HyperNetwork, CNFRenderer
from src.models_v22 import HyperNetwork_Perceiver_v22, GaborRenderer_v22, GaborRenderer_v22_alter, HyperNetwork_Perceiver_v23, GaborRenderer_v23
from src.models_v2 import HyperNetwork_Perceiver_v5, GaborRenderer_v5, HyperNetwork_Perceiver_v55, GaborRenderer_v55
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
    SHUFFLE_NODE_ORDER = getattr(hp, "shuffle_node_order", True)
    SHUFFLE_NODE_SEED = getattr(hp, "shuffle_node_seed", None)
    
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
        iter_dataset = iter(dataset)
        for i in range(5):
            original_coords_sample, mesh_info = next(iter_dataset)
        original_coords = original_coords_sample.unsqueeze(0).clone().detach().to(device) # expand to [1, N, 2]
        cells_tensor = mesh_info['cells']
        gt_fields_tensor = mesh_info['fields'].unsqueeze(0).clone().detach() # shape [1, T_CHUNK, N, C]
        if USE_NODE_TYPE:
            node_type_tensor = mesh_info['node_type'].unsqueeze(0).clone().detach().to(device) # shape [1, N, 1]
        else:
            node_type_tensor = None

        if SHUFFLE_NODE_ORDER:
            num_nodes = original_coords.shape[1]
            shuffle_gen = torch.Generator(device="cpu")
            if SHUFFLE_NODE_SEED is not None:
                shuffle_gen.manual_seed(int(SHUFFLE_NODE_SEED))
                print(f"Shuffling node order with fixed seed: {int(SHUFFLE_NODE_SEED)}")
            else:
                print("Shuffling node order with random seed.")

            # Keep the same permutation across all node-indexed tensors.
            perm_cpu = torch.randperm(num_nodes, generator=shuffle_gen)
            inv_perm_cpu = torch.empty_like(perm_cpu)
            inv_perm_cpu[perm_cpu] = torch.arange(num_nodes)

            perm_coords = perm_cpu.to(original_coords.device)
            perm_fields = perm_cpu.to(gt_fields_tensor.device)
            original_coords = original_coords[:, perm_coords, :]
            gt_fields_tensor = gt_fields_tensor[:, :, perm_fields, :]
            if node_type_tensor is not None:
                perm_node_type = perm_cpu.to(node_type_tensor.device)
                node_type_tensor = node_type_tensor[:, perm_node_type, :]

            # Remap mesh connectivity so triangles still point to the right shuffled nodes.
            cells_long = cells_tensor.long()
            valid_mask = cells_long >= 0
            remapped_cells = cells_long.clone()
            remapped_cells[valid_mask] = inv_perm_cpu[cells_long[valid_mask].cpu()].to(remapped_cells.device)
            cells_tensor = remapped_cells.to(cells_tensor.dtype)

            print(f"Applied node permutation for {num_nodes} nodes. First 10 perm indices: {perm_cpu[:10].tolist()}")
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
    elif ENCODER_TYPE == "HyperNetwork_Perceiver_v5":
        print("Using Perceiver_v5-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v5(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
            use_node_type=USE_NODE_TYPE
        ).to(device)
    elif ENCODER_TYPE == "HyperNetwork_Perceiver_v55":
        print("Using Perceiver_v55-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v55(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
            use_node_type=USE_NODE_TYPE
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
    elif RENDERER_TYPE == "GaborRenderer_v22_alter":
        print("Using Perceiver_v22-based GaborRenderer_alter")
        cnf = GaborRenderer_v22_alter(
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
    elif RENDERER_TYPE == "GaborRenderer_v5":
        print("Using Perceiver_v5-based GaborRenderer")
        cnf = GaborRenderer_v5(
            latent_dim=LATENT_DIM, 
            coord_dim=2, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF, 
            use_node_type=USE_NODE_TYPE
        ).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v55":
        print("Using Perceiver_v55-based GaborRenderer")
        cnf = GaborRenderer_v55(
            latent_dim=LATENT_DIM, 
            coord_dim=2, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF, 
            use_node_type=USE_NODE_TYPE
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
            num_steps = getattr(hp, "num_sampling_steps", 5)
            print(f"Mode: Multi-step consistency sampling ({num_steps} steps)")

        t_max = getattr(hp, "t_max", 80.0)
        t_min = getattr(hp, "t_min", 0.003)
        rho = 3.0
        
        # EDM target timestep schedule (Karras et al. 2022)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        t_steps = (t_max ** (1 / rho) + step_indices / (max(num_steps - 1, 1)) * (t_min ** (1 / rho) - t_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # Append t=0 for the final step condition
        
        x = torch.randn(B, T, N, C).to(device) * t_max
        coords = original_coords # [B, N, 2]
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
            if USE_NODE_TYPE:
                z1_gen = encoder(c_in * x, coords_norm, t_target, node_type_tensor)
            else:
                z1_gen = encoder(c_in * x, coords_norm, t_target)
            
            # 5. Render fields
            if USE_NODE_TYPE:
                f_theta = cnf(z1_gen, coords_norm, node_type_tensor) # [1, T_CHUNK, N, C_OUT]
            else:
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
        field_pred_np = trajectory_pred.detach().cpu().numpy()[0]
        field_gt_np = gt_fields_tensor.cpu().numpy()[0]
        T_len = field_pred_np.shape[0]
        num_channels = field_pred_np.shape[2]
        
        fig_freq, ax_freq = plt.subplots(1, 3, figsize=(18, 5))

        # 0. Pred vs GT total energy comparison over time
        def compute_energy_curve(field_tnc):
            # field_tnc: [T, N, C]
            if field_tnc.shape[2] >= 2:
                # Kinetic energy proxy from velocity channels (u, v): E = 0.5 * (u^2 + v^2)
                energy_node = 0.5 * np.sum(field_tnc[:, :, :2] ** 2, axis=2)
            else:
                energy_node = 0.5 * np.sum(field_tnc ** 2, axis=2)
            return np.mean(energy_node, axis=1)

        energy_curve_pred = compute_energy_curve(field_pred_np)
        energy_curve_gt = compute_energy_curve(field_gt_np)
        energy_abs_err = np.abs(energy_curve_pred - energy_curve_gt)
        energy_mae = float(np.mean(energy_abs_err))
        energy_rel_l2 = float(np.linalg.norm(energy_curve_pred - energy_curve_gt) / (np.linalg.norm(energy_curve_gt) + 1e-12))

        ax_freq[2].plot(np.arange(T_len), energy_curve_gt, marker='x', linestyle='--', label='GT Energy')
        ax_freq[2].plot(np.arange(T_len), energy_curve_pred, marker='o', linestyle='-', label='Pred Energy')
        ax_freq[2].fill_between(
            np.arange(T_len),
            np.minimum(energy_curve_gt, energy_curve_pred),
            np.maximum(energy_curve_gt, energy_curve_pred),
            alpha=0.2,
            color='gray',
            label='Abs Diff'
        )
        ax_freq[2].set_xlabel('Time step')
        ax_freq[2].set_ylabel('Average Energy')
        ax_freq[2].set_title('Pred vs GT Energy (Time Domain)')
        ax_freq[2].grid(True, ls='--', alpha=0.5)
        ax_freq[2].legend()
        ax_freq[2].text(
            0.02,
            0.98,
            f'MAE={energy_mae:.4e}\nRelL2={energy_rel_l2:.4e}',
            transform=ax_freq[2].transAxes,
            va='top',
            ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        print(f'Energy comparison: MAE={energy_mae:.6e}, RelL2={energy_rel_l2:.6e}')
        
        # 1. 时间轴频域能量 (Temporal Frequency Energy)
        # 对时间轴进行 1D FFT [T_CHUNK, N, C]
        fft_temporal_pred = np.fft.fft(field_pred_np, axis=0)
        fft_temporal_gt = np.fft.fft(field_gt_np, axis=0)
        energy_temporal_pred = np.abs(fft_temporal_pred) ** 2
        energy_temporal_gt = np.abs(fft_temporal_gt) ** 2
        
        # 计算平均能量并取出正频率部分
        freqs = np.fft.fftfreq(T_len)
        pos_freq_idxs = freqs > 0
        
        # 将频率对应为波数 (Wavenumber/Mode k)
        k_temporal = np.arange(1, np.sum(pos_freq_idxs) + 1)
        
        # 仅在 N 维度求平均，保留通道维度 C: 输出形状 [频率数, 通道数]
        energy_temporal_avg_pred = np.mean(energy_temporal_pred, axis=1)[pos_freq_idxs]
        energy_temporal_avg_gt = np.mean(energy_temporal_gt, axis=1)[pos_freq_idxs]
        
        for c in range(num_channels):
            ax_freq[0].plot(k_temporal, energy_temporal_avg_gt[:, c], marker='x', linestyle='--', label=f"GT Ch{c}")
            ax_freq[0].plot(k_temporal, energy_temporal_avg_pred[:, c], marker='o', linestyle='-', label=f"Pred Ch{c}")
            
        ax_freq[0].set_yscale('log')
        # ax_freq[0].set_xscale('log')
        ax_freq[0].set_xlabel("Temporal Wavenumber ($k$)")
        ax_freq[0].set_ylabel("Average Energy")
        ax_freq[0].set_title("Temporal Energy Spectrum (GT vs Pred)")
        ax_freq[0].grid(True, which="both", ls="--", alpha=0.5)
        ax_freq[0].legend()

        # 2. 空间轴频域能量 (Spatial Frequency Energy / Wavenumber Spectrum)
        # 取定初始时刻 (t=0) 的空间场通过插值为网格进行 2D FFT 估计物理场的能量级联
        try:
            from scipy.interpolate import griddata
            
            data0_pred = field_pred_np[0] # [N, C]
            data0_gt = field_gt_np[0] # [N, C]
            x_coords = coord[:, 0]
            y_coords = coord[:, 1]
            
            # 插值到 128x128 的规则网格
            grid_res = 128
            grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):complex(0, grid_res), min(y_coords):max(y_coords):complex(0, grid_res)]
            
            for c in range(num_channels):
                values_pred_c = data0_pred[:, c]
                values_gt_c = data0_gt[:, c]

                grid_pred = griddata((x_coords, y_coords), values_pred_c, (grid_x, grid_y), method='linear')
                grid_gt = griddata((x_coords, y_coords), values_gt_c, (grid_x, grid_y), method='linear')
                grid_pred[np.isnan(grid_pred)] = 0.0 # 填补 nan
                grid_gt[np.isnan(grid_gt)] = 0.0 # 填补 nan

                # 进行 2D FFT 并将其中心化
                fft2_spatial_pred = np.fft.fft2(grid_pred)
                fft2_spatial_gt = np.fft.fft2(grid_gt)
                energy_spatial_pred_2d = np.abs(np.fft.fftshift(fft2_spatial_pred)) ** 2
                energy_spatial_gt_2d = np.abs(np.fft.fftshift(fft2_spatial_gt)) ** 2

                # 进行径向平均(Radial average)获得各波数(wavenumber)频段的一维能量谱
                center_x, center_y = grid_res // 2, grid_res // 2
                y_idx, x_idx = np.indices(grid_pred.shape)
                r = np.sqrt((x_idx - center_x) ** 2 + (y_idx - center_y) ** 2).astype(int)

                tbin_pred = np.bincount(r.ravel(), energy_spatial_pred_2d.ravel())
                tbin_gt = np.bincount(r.ravel(), energy_spatial_gt_2d.ravel())
                nr = np.bincount(r.ravel())
                radialprofile_pred = tbin_pred / np.maximum(nr, 1)
                radialprofile_gt = tbin_gt / np.maximum(nr, 1)

                max_len = min(len(radialprofile_pred), len(radialprofile_gt))
                k_vals = np.arange(1, max_len) # 略去 k=0 的直流分量
                ax_freq[1].plot(k_vals, radialprofile_gt[1:max_len], marker='x', linestyle='--', label=f"GT Ch{c}")
                ax_freq[1].plot(k_vals, radialprofile_pred[1:max_len], marker='.', linestyle='-', label=f"Pred Ch{c}")
                
            ax_freq[1].set_yscale('log')
            # ax_freq[1].set_xscale('log')
            ax_freq[1].set_xlabel("Wavenumber ($k$)")
            ax_freq[1].set_ylabel("Energy $E(k)$")
            ax_freq[1].set_title("Spatial Energy Spectrum (t=0, GT vs Pred)")
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
