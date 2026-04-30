# self-distillation training script 

import sys
import os
import copy
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basicutility import ReadInput as ri
from src.dataset2 import NpyChunkDataset
# from src.models import HyperNetwork, CNFRenderer
from src.models_ae import HyperNetwork_GINO, GaborRenderer_GINO
from src.models_v22 import HyperNetwork_Perceiver_v22, GaborRenderer_v22, HyperNetwork_Perceiver_v23, GaborRenderer_v23
from src.normalize import Normalizer_ts, compute_dataset_statistics
from src.utils import *


def _unwrap_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _parse_gpu_ids(gpu_ids_cfg, num_available):
    if gpu_ids_cfg is None:
        return list(range(num_available))
    if isinstance(gpu_ids_cfg, str):
        ids = [int(x.strip()) for x in gpu_ids_cfg.split(",") if x.strip()]
    elif isinstance(gpu_ids_cfg, (list, tuple)):
        ids = [int(x) for x in gpu_ids_cfg]
    else:
        ids = [int(gpu_ids_cfg)]
    ids = [i for i in ids if 0 <= i < num_available]
    return ids


def _load_model_state_dict(model, state_dict):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def _find_latest_checkpoint(save_path):
    patterns = [
        os.path.join(save_path, "checkpoint_step_*.pt"),
        os.path.join(save_path, "checkpoint_epoch_*.pt"),
        os.path.join(save_path, "checkpoint_final.pt"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob(p))
    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _ema_update(ema_model, online_model, decay):
    online_state = _unwrap_state_dict(online_model)
    with torch.no_grad():
        for k, ema_v in ema_model.state_dict().items():
            online_v = online_state[k].detach()
            if torch.is_floating_point(ema_v):
                ema_v.mul_(decay).add_(online_v, alpha=(1.0 - decay))
            else:
                ema_v.copy_(online_v)


def _estimate_steps_per_epoch(dataset, batch_size, steps_override=None):
    """
    Estimate per-epoch optimizer steps for IterableDataset without relying on len(dataloader).
    """
    if steps_override is not None:
        try:
            steps = int(steps_override)
            if steps > 0:
                return steps
        except Exception:
            pass

    num_sims = getattr(dataset, "num_sims", None)
    shape_t = getattr(dataset, "shape_t", None)
    chunk_size = getattr(dataset, "chunk_size", None)
    stride = getattr(dataset, "stride", None)

    if (
        isinstance(num_sims, int)
        and isinstance(shape_t, int)
        and isinstance(chunk_size, int)
        and isinstance(stride, int)
        and num_sims > 0
        and shape_t >= chunk_size
        and stride > 0
    ):
        chunks_per_sim = ((shape_t - chunk_size) // stride) + 1
        total_samples = num_sims * chunks_per_sim
        return max(1, (total_samples + batch_size - 1) // batch_size)

    print("Warning: failed to infer steps_per_epoch for IterableDataset, fallback to 1. Set hp.steps_per_epoch for accuracy.")
    return 1

def train(hp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Configuration Settings
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
    EPOCHS = getattr(hp, "epochs", 50)
    BATCH_SIZE = getattr(hp, "batch_size", 32)
    T_SAMPLING = getattr(hp, "t_sampling", "edm")
    T_BETA_ALPHA = getattr(hp, "t_beta_alpha", 1.0)
    T_BETA_BETA = getattr(hp, "t_beta_beta", 3.0)
    LR_enc = getattr(hp, "lr_enc", 1e-4)
    LR_cnf = getattr(hp, "lr_cnf", 1e-4)
    STRIDE = getattr(hp, "stride", T_CHUNK)
    SAVE_PATH = getattr(hp, "save_path", "saved_models")
    USE_MULTI_GPU = getattr(hp, "use_multi_gpu", False)
    GPU_IDS_CFG = getattr(hp, "gpu_ids", None)
    SAVE_EVERY_EPOCHS = getattr(hp, "save_every_epochs", 10)
    SAVE_EVERY_STEPS = getattr(hp, "save_every_steps", 0)  # 0 to disable step-level saving
    DISTILL_LAMBDA = getattr(hp, "distill_lambda", 1.0)

    EMA_DECAY = getattr(hp, "ema_decay", 0.999)

    AUTO_RESUME = getattr(hp, "auto_resume", True)
    RESUME_CHECKPOINT = getattr(hp, "resume_checkpoint", "latest")
    LOSS_TYPE = getattr(hp, "loss_type", "MSE").upper()
    
    PHASE1_EPOCHS = getattr(hp, "phase1_epochs", EPOCHS // 2)
    T_MIN = getattr(hp, "t_min", 0.003)
    T_MAX = getattr(hp, "t_max", 80.0)
    T_EPS1 = getattr(hp, "t_eps1", 0.007)
    T_EPS2 = getattr(hp, "t_eps2", 0.0865)
    LAMBDA_GENE = getattr(hp, "lambda_gene", DISTILL_LAMBDA)
    LAMBDA_RECON = getattr(hp, "lambda_recon", 1.0)
    LAMBDA_DIFF = getattr(hp, "lambda_diff", 1.0)
    
    # Normalizer configs
    norm_cfg = getattr(hp, "normalizer", {})
    COORD_METHOD = norm_cfg.get("coord_method", "-11") if norm_cfg else "-11"
    FIELD_METHOD = norm_cfg.get("field_method", "ms") if norm_cfg else "ms"
    COORD_DIM = norm_cfg.get("coord_dim", None) if norm_cfg else None
    FIELD_DIM = norm_cfg.get("field_dim", None) if norm_cfg else None

    os.makedirs(SAVE_PATH, exist_ok=True)
    NORM_PARAMS_PATH = os.path.join(SAVE_PATH, "normalizer_params.pt")
    
    # Tensorboard writer
    log_dir = os.path.join(SAVE_PATH, "tensorboard_logs")
    writer = SummaryWriter(log_dir=log_dir)

    # 2. Build Dataset & DataLoader
    # (Wrapped in try/except so if dataset path is missing, users know what to edit)
    try:
        # from src.dataset2 import NpyChunkDataset
        import numpy as np
        
        dataset = NpyChunkDataset(
            dataset_path=DATASET_PATH,
            chunk_size=T_CHUNK,
            stride=STRIDE,
            mode='train'
        )
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=getattr(hp, "num_workers", 4))
        STEPS_PER_EPOCH = _estimate_steps_per_epoch(
            dataset,
            BATCH_SIZE,
            steps_override=getattr(hp, "steps_per_epoch", None),
        )
        print(f"Estimated steps_per_epoch={STEPS_PER_EPOCH} for phase-2 schedule.")
    except Exception as e:
        print(f"Failed to load dataset: {e}. Please ensure DATASET_PATH is correct.")
        return

    # 2.5 Setup Normalizers
    os.makedirs("saved_models", exist_ok=True)
    coord_params, field_params = None, None
    if os.path.exists(NORM_PARAMS_PATH):
        print(f"Loading normalizer parameters from {NORM_PARAMS_PATH}")
        params = torch.load(NORM_PARAMS_PATH)
        coord_params = params['coord_params']
        field_params = params['field_params']
    else:
        print("Normalizer params not found. Computing from dataset...")
        coord_params, field_params = compute_dataset_statistics(
            dataset, 
            coord_method=COORD_METHOD, 
            field_method=FIELD_METHOD,
            coord_dim=COORD_DIM,
            field_dim=FIELD_DIM
        )
        torch.save({
            'coord_params': coord_params,
            'field_params': field_params
        }, NORM_PARAMS_PATH)
        
    coord_normalizer = Normalizer_ts(params=coord_params, method=COORD_METHOD, dim=COORD_DIM)
    field_normalizer = Normalizer_ts(params=field_params, method=FIELD_METHOD, dim=FIELD_DIM)

    # 3. Initialize Unified Architecture
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
        print("Using Perceiver_v23-based HyperNetwork with improved time conditioning")
        encoder = HyperNetwork_Perceiver_v23(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
            use_node_type=getattr(hp, "use_node_type", False)
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
        ## Error
        raise ValueError(f"Unsupported ENCODER_TYPE: {ENCODER_TYPE}")

    count_parameters(encoder, name="Encoder")

    # EMA teacher keeps a historical weighted average of the online encoder.
    encoder_ema = copy.deepcopy(encoder).to(device)
    encoder_ema.eval()
    for p in encoder_ema.parameters():
        p.requires_grad_(False)
    
    if RENDERER_TYPE == "GaborRenderer_v22":
        print("Using GaborRenderer_v22")
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
        print("Using GaborRenderer_v23 with improved conditioning and stability")
        cnf = GaborRenderer_v23(
            latent_dim=LATENT_DIM, 
            coord_dim=2, 
            t_chunk=T_CHUNK, 
            channel_out=C_OUT, 
            hidden_dim=HIDDEN_DIM, 
            num_layers=NUM_LAYERS_CNF,
            use_node_type=getattr(hp, "use_node_type", False)
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
        ## Error
        raise ValueError(f"Unsupported RENDERER_TYPE: {RENDERER_TYPE}")

    count_parameters(cnf, name="CNF")

    # EMA teacher for CNF
    cnf_ema = copy.deepcopy(cnf).to(device)
    cnf_ema.eval()
    for p in cnf_ema.parameters():
        p.requires_grad_(False)

    # Optional multi-GPU support via DataParallel
    if USE_MULTI_GPU and device.type == "cuda":
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            device_ids = _parse_gpu_ids(GPU_IDS_CFG, n_gpu)
            if len(device_ids) > 1:
                encoder = nn.DataParallel(encoder, device_ids=device_ids)
                cnf = nn.DataParallel(cnf, device_ids=device_ids)
                print(f"Multi-GPU enabled (DataParallel), device_ids={device_ids}")
            else:
                print(f"use_multi_gpu=True but valid gpu_ids={device_ids}, fallback to single GPU training")
        else:
            print("use_multi_gpu=True but less than 2 CUDA devices found, fallback to single GPU training")
    
    # 3.5 Optimizers and Schedulers
    optimizer_encoder = torch.optim.AdamW(encoder.parameters(), lr=LR_enc, weight_decay=1e-4)
    optimizer_cnf = torch.optim.AdamW(cnf.parameters(), lr=LR_cnf, weight_decay=1e-4)

    WARMUP_EPOCHS = getattr(hp, "warmup_epochs", max(1, int(EPOCHS * 0.1)))
    MIN_LR = getattr(hp, "min_lr", 1e-6)
    COSINE_EPOCHS = getattr(hp, "cosine_epochs", EPOCHS - WARMUP_EPOCHS)
    
    # Schedulers for encoder
    warmup_enc = torch.optim.lr_scheduler.LinearLR(optimizer_encoder, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_enc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=COSINE_EPOCHS, eta_min=MIN_LR)
    constant_enc = torch.optim.lr_scheduler.ConstantLR(optimizer_encoder, factor=MIN_LR / LR_enc, total_iters=EPOCHS)
    scheduler_encoder = torch.optim.lr_scheduler.SequentialLR(
        optimizer_encoder, 
        schedulers=[warmup_enc, cosine_enc, constant_enc], 
        milestones=[WARMUP_EPOCHS, WARMUP_EPOCHS + COSINE_EPOCHS]
    )

    # Schedulers for cnf
    warmup_cnf = torch.optim.lr_scheduler.LinearLR(optimizer_cnf, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_cnf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnf, T_max=COSINE_EPOCHS, eta_min=MIN_LR)
    constant_cnf = torch.optim.lr_scheduler.ConstantLR(optimizer_cnf, factor=MIN_LR / LR_cnf, total_iters=EPOCHS)
    scheduler_cnf = torch.optim.lr_scheduler.SequentialLR(
        optimizer_cnf, 
        schedulers=[warmup_cnf, cosine_cnf, constant_cnf], 
        milestones=[WARMUP_EPOCHS, WARMUP_EPOCHS + COSINE_EPOCHS]
    )

    # 3.6 Optionally resume from the latest/specified checkpoint
    global_step = 0
    start_epoch = 0
    ckpt_path = None
    if AUTO_RESUME:
        if isinstance(RESUME_CHECKPOINT, str) and RESUME_CHECKPOINT.lower() == "latest":
            ckpt_path = _find_latest_checkpoint(SAVE_PATH)
        elif isinstance(RESUME_CHECKPOINT, str) and RESUME_CHECKPOINT.strip():
            ckpt_path = RESUME_CHECKPOINT

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        _load_model_state_dict(encoder, ckpt['encoder_state_dict'])
        _load_model_state_dict(cnf, ckpt['cnf_state_dict'])
        if 'encoder_ema_state_dict' in ckpt:
            encoder_ema.load_state_dict(ckpt['encoder_ema_state_dict'])
        else:
            _load_model_state_dict(encoder_ema, _unwrap_state_dict(encoder))
            print("Checkpoint has no encoder_ema_state_dict, EMA initialized from online encoder.")
            
        if 'cnf_ema_state_dict' in ckpt:
            cnf_ema.load_state_dict(ckpt['cnf_ema_state_dict'])
        else:
            _load_model_state_dict(cnf_ema, _unwrap_state_dict(cnf))
            print("Checkpoint has no cnf_ema_state_dict, EMA initialized from online CNF.")
            
        optimizer_encoder.load_state_dict(ckpt['optimizer_encoder_state_dict'])
        optimizer_cnf.load_state_dict(ckpt['optimizer_cnf_state_dict'])
        scheduler_encoder.load_state_dict(ckpt['scheduler_encoder_state_dict'])
        scheduler_cnf.load_state_dict(ckpt['scheduler_cnf_state_dict'])

        global_step = int(ckpt.get('global_step', 0))
        start_epoch = int(ckpt.get('epoch', 0))
        print(f"Resume state: start_epoch={start_epoch}, global_step={global_step}")
    elif AUTO_RESUME:
        print("No checkpoint found for auto_resume, training from scratch.")

    # 4. Training Loop (End-to-End INRCT styled Flow Matching)
    print("Starting End-to-End Training (Data-Space Supervision + Self-Distillation) ...")
    
    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc="Training Epochs")
    for epoch in epoch_pbar:
        encoder.train()
        cnf.train()
        encoder_ema.eval()
        cnf_ema.eval()

        dataset.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_gene_loss = 0.0
        epoch_recon_unweighted = 0.0
        epoch_gene_unweighted = 0.0
        epoch_gnorm_enc = 0.0
        epoch_gnorm_cnf = 0.0
        recon_steps = 0
        epoch_ratio_sum = 0.0
        epoch_ratio_steps = 0
        epoch_ratio_min = float("inf")
        epoch_ratio_max = float("-inf")
        
        loss_t_sums = { '1.0': 0.0, '0.75': 0.0, '0.5': 0.0, '0.25': 0.0, '0.0': 0.0 }
        loss_t_unweight_sums = { '1.0': 0.0, '0.75': 0.0, '0.5': 0.0, '0.25': 0.0, '0.0': 0.0 }
        loss_t_counts = { '1.0': 0, '0.75': 0, '0.5': 0, '0.25': 0, '0.0': 0 }
        
        for step, (coords_batch, traj_batch) in enumerate(dataloader):
            # coords_batch: [B, N, 2]
            # traj_batch:   [B, T, N, C]
            coords = coords_batch.to(device)
            x_real = traj_batch.to(device)
            B = x_real.shape[0]
            
            # ===== 0. Normalize inputs =====
            coords = coord_normalizer.normalize(coords)
            x_real = field_normalizer.normalize(x_real)

            # ===== A. Common Setup =====
            optimizer_encoder.zero_grad()
            optimizer_cnf.zero_grad()
            
            # Sample Pure Noise matching physical shape
            noise = torch.randn_like(x_real)
            
            # --- Random Point Drop for Zero-Shot & Unconditional Support ---
            N_pts = coords.shape[1]
            keep_ratio = torch.rand(1).item() * 0.6 + 0.4 # Keep 40% to 100% points
            M_pts = max(1, int(N_pts * keep_ratio))
            
            indices = torch.randperm(N_pts)[:M_pts].to(device)
            coords_obs = coords[:, indices, :]
            coords_query = coords
            
            def compute_element_loss(pred, target):
                if LOSS_TYPE in ["L1", "MAE"]:
                    return F.l1_loss(pred, target, reduction='none')
                elif LOSS_TYPE == "HUBER":
                    return F.huber_loss(pred, target, reduction='none')
                else:
                    return F.mse_loss(pred, target, reduction='none')

            # ====================================================
            # PHASE 1: Reconstruction Training (Small noise only)
            # ====================================================
            if epoch < PHASE1_EPOCHS:
                t = torch.rand(B, device=device) * (T_EPS1 - T_MIN) + T_MIN
                t = torch.clamp(t, min=T_MIN, max=T_MAX) # Ensure t is within reasonable bounds
                t_expand = t.view(B, 1, 1, 1)
                
                # Create Noisy Trajectory (EDM: x_t = x_real + t * noise)
                x_noisy = x_real + t_expand * noise
                x_noisy_obs = x_noisy[:, :, indices, :]
                
                # EDM-style Skip Connection Scaling Factors
                sigma = t_expand
                sigma_data = 0.5
                c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data**2) # Pre-conditioning
                
                # 1. Predict clean dynamic latent Z1 from noisy data and coords 
                z1_pred = encoder(c_in * x_noisy_obs, coords_obs, t) # [B, LATENT_DIM]

                # 2. Render trajectory directly using Z1 and spatial coordinates
                f_theta = cnf(z1_pred, coords_query) # [B, T_CHUNK, N, C]
                
                # ===== C. Data-Space Supervision =====
                # small_noise_mask1 = (t < T_EPS1).float().view(B, 1, 1, 1)

                loss_weight = 1.0 / torch.sqrt(t ** 2 + sigma_data**2) # Higher weight for smaller t, similar to EDM's weighting strategy

                # L_recon: At small noise, force f_theta to learn gt
                l_recon_unweighted = compute_element_loss(f_theta, x_real)
                recon_loss = (l_recon_unweighted.reshape(B, -1).mean(dim=1) * loss_weight).mean()

                # Stage 1: Only optimize L_recon
                loss = recon_loss
                gene_loss = torch.tensor(0.0, device=device)
                
                # For logging bin tracking
                l_gene_unweighted = torch.zeros_like(l_recon_unweighted)

            # ====================================================
            # PHASE 2: Self-Distillation & Generation Training
            # ====================================================
            else:
                # Full noise range
                if T_SAMPLING == "beta":
                    m = torch.distributions.beta.Beta(torch.tensor([T_BETA_ALPHA]), torch.tensor([T_BETA_BETA]))
                    t = m.sample((B,)).squeeze(-1).to(device) * 80.0
                elif T_SAMPLING == "uniform":
                    t = torch.rand(B, device=device) * 80.0
                else: # edm default
                    P_mean, P_std = -1.1, 2.0
                    t = (torch.randn(B, device=device) * P_std + P_mean).exp()
            
                # Expand t to match image dims: [B, 1, 1, 1]
                t = torch.clamp(t, min=T_MIN, max=T_MAX) # Ensure t is within reasonable bounds
                t_expand = t.view(B, 1, 1, 1)
                
                # Create Noisy Trajectory (EDM: x_t = x_real + t * noise)
                x_noisy = x_real + t_expand * noise
                x_noisy_obs = x_noisy[:, :, indices, :]
                x_noisy_query = x_noisy
                
                # In phase 2, use structured sampling strategy for distillation
                k_val = getattr(hp, "teacher_k", 8.0)
                b_val = getattr(hp, "teacher_b", 1.0)
                q_val = getattr(hp, "teacher_q", 2.0)
                d_val = getattr(hp, "teacher_d", 5000.0)
                
                phase2_step = (epoch - PHASE1_EPOCHS) * STEPS_PER_EPOCH + step
                n_t = 1.0 + k_val * torch.sigmoid(-b_val * t)
                ratio = 1.0 - (1.0 / (q_val ** (phase2_step / d_val))) * n_t
                ratio = torch.clamp(ratio, min=0.0, max=1.0)
                ratio_mean_batch = ratio.mean().item()
                epoch_ratio_sum += ratio_mean_batch
                epoch_ratio_steps += 1
                epoch_ratio_min = min(epoch_ratio_min, ratio.min().item())
                epoch_ratio_max = max(epoch_ratio_max, ratio.max().item())
                t_teacher = t * ratio

                t_teacher = torch.clamp(t_teacher, min=T_MIN, max=T_MAX) # Ensure t_teacher is within reasonable bounds
                t_teacher_expand = t_teacher.view(B, 1, 1, 1)
                x_noisy_teacher = x_real + t_teacher_expand * noise
                x_noisy_teacher_obs = x_noisy_teacher[:, :, indices, :]

                # EDM-style Skip Connection Scaling Factors
                sigma = t_expand
                sigma_data = 0.5
                c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data**2) # Pre-conditioning
                
                # 1. Predict clean dynamic latent Z1 from noisy data and coords 
                z1_pred = encoder(c_in * x_noisy_obs, coords_obs, t) # [B, LATENT_DIM]

                # 2. Render trajectory directly using Z1 and spatial coordinates
                f_theta = cnf(z1_pred, coords_query) # [B, T_CHUNK, N, C]
                
                # Direct prediction
                x_pred = f_theta

                with torch.no_grad():
                    c_in_teach = 1.0 / torch.sqrt(t_teacher_expand**2 + sigma_data**2)
                    z_target = encoder_ema(c_in_teach * x_noisy_teacher_obs, coords_obs, t_teacher).detach()
                    f_theta_teacher = cnf_ema(z_target, coords_query).detach()
                    
                    x_pred_teacher = f_theta_teacher

                # ===== C. Data-Space Supervision =====
                loss_weight = 1.0 / torch.sqrt(t ** 2 + sigma_data**2) # Higher weight for smaller t, similar to EDM's weighting strategy, shape: [B]
                 
                delta = x_pred - x_pred_teacher
                delta_norm = delta.pow(2).mean(dim=[1,2,3], keepdim=True).detach() # shape [B, 1, 1, 1]
                loss_weight_delta = 1.0 / torch.sqrt(delta_norm + (1.0e-2) ** 2) # shape [B, 1, 1, 1], higher weight for samples where student is far from teacher

                small_noise_mask1 = (t < T_EPS2).float().view(B, 1, 1, 1)

                # L_recon: At small noise, force f_theta to learn gt
                l_recon_unweighted = compute_element_loss(f_theta, x_real) * small_noise_mask1
                recon_loss = (l_recon_unweighted.reshape(B, -1).mean(dim=1) * loss_weight).sum() / (small_noise_mask1.sum() + 1e-8)

                # L_gene: Generation consistency loss between online and teacher
                l_gene_unweighted = compute_element_loss(x_pred, x_pred_teacher)
                gene_loss = (l_gene_unweighted.reshape(B, -1).mean(dim=1) * loss_weight * loss_weight_delta.squeeze()).mean()

                # Stage 2: Train generation with full noise range and other losses
                loss = LAMBDA_GENE * gene_loss + LAMBDA_RECON * recon_loss
            
            # Backprop updates BOTH the HyperNetwork and the CNF at the same time
            loss.backward()
            
            # Compute gradient norm
            gnorm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            gnorm_cnf = torch.nn.utils.clip_grad_norm_(cnf.parameters(), max_norm=5.0)
            
            optimizer_encoder.step()
            optimizer_cnf.step()

            _ema_update(encoder_ema, encoder, EMA_DECAY)
            _ema_update(cnf_ema, cnf, EMA_DECAY)
            
            epoch_loss += loss.item()
            epoch_gene_loss += gene_loss.item()
            epoch_gnorm_enc += gnorm_enc.item() if isinstance(gnorm_enc, torch.Tensor) else gnorm_enc
            epoch_gnorm_cnf += gnorm_cnf.item() if isinstance(gnorm_cnf, torch.Tensor) else gnorm_cnf
            
            global_step += 1
            
            # Keep checkpoint every x steps if enabled
            if SAVE_EVERY_STEPS > 0 and global_step % SAVE_EVERY_STEPS == 0:
                step_ckpt = os.path.join(SAVE_PATH, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'encoder_state_dict': _unwrap_state_dict(encoder),
                    'encoder_ema_state_dict': encoder_ema.state_dict(),
                    'cnf_state_dict': _unwrap_state_dict(cnf),
                    'cnf_ema_state_dict': cnf_ema.state_dict(),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'optimizer_cnf_state_dict': optimizer_cnf.state_dict(),
                    'scheduler_encoder_state_dict': scheduler_encoder.state_dict(),
                    'scheduler_cnf_state_dict': scheduler_cnf.state_dict(),
                }, step_ckpt)
            
            # Bin the sample losses based on t value
            
            # Using loss for tracking, you can use any subset of the loss
            recon_b = l_recon_unweighted.reshape(B, -1).mean(dim=1)
            gene_b = l_gene_unweighted.reshape(B, -1).mean(dim=1)

            if epoch < PHASE1_EPOCHS:
                valid_recon_mask = torch.ones(B, device=device)
                sample_loss_val_reduced = (recon_b * loss_weight.reshape(-1)).detach()
                sample_loss_val_unweighted = recon_b.detach()
            else:
                valid_recon_mask = (t < T_EPS2).float()
                # Use identical loss reduction formulas to the objective
                loss_weight_flat = loss_weight.reshape(-1)
                loss_weight_delta_flat = loss_weight_delta.reshape(-1)
                sample_loss_val_reduced = (LAMBDA_GENE * gene_b * loss_weight_flat * loss_weight_delta_flat + 
                                           LAMBDA_RECON * recon_b * loss_weight_flat).detach()
                sample_loss_val_unweighted = (LAMBDA_GENE * gene_b + 
                                           LAMBDA_RECON * recon_b).detach()
                                           
            # Add to total unweighted, compute only over effectively handled sizes
            valid_count = valid_recon_mask.sum().item()
            if valid_count > 0:
                epoch_recon_unweighted += (recon_b.sum().item() / valid_count)
                epoch_recon_loss += recon_loss.item()
                recon_steps += 1
            
            epoch_gene_unweighted += gene_b.mean().item()

            for i in range(B):
                ti = t[i].item()
                val = sample_loss_val_reduced[i].item()
                val_uw = sample_loss_val_unweighted[i].item()
                
                # EDM has t scale from 0 to 80+
                if ti > 20.0:
                    loss_t_sums['1.0'] += val
                    loss_t_unweight_sums['1.0'] += val_uw
                    loss_t_counts['1.0'] += 1
                elif ti > 5.0:
                    loss_t_sums['0.75'] += val
                    loss_t_unweight_sums['0.75'] += val_uw
                    loss_t_counts['0.75'] += 1
                elif ti > 1.0:
                    loss_t_sums['0.5'] += val
                    loss_t_unweight_sums['0.5'] += val_uw
                    loss_t_counts['0.5'] += 1
                elif ti > 0.2:
                    loss_t_sums['0.25'] += val
                    loss_t_unweight_sums['0.25'] += val_uw
                    loss_t_counts['0.25'] += 1
                else:
                    loss_t_sums['0.0'] += val
                    loss_t_unweight_sums['0.0'] += val_uw
                    loss_t_counts['0.0'] += 1

            # Remove inner pbar postfix update
            
        avg_loss = epoch_loss / (step + 1)
        avg_recon_loss = epoch_recon_loss / max(recon_steps, 1)
        avg_gene_loss = epoch_gene_loss / (step + 1)
        avg_recon_unweighted = epoch_recon_unweighted / max(recon_steps, 1)
        avg_gene_unweighted = epoch_gene_unweighted / (step + 1)
        avg_gnorm_enc = epoch_gnorm_enc / (step + 1)
        avg_gnorm_cnf = epoch_gnorm_cnf / (step + 1)
        if epoch_ratio_steps > 0:
            avg_ratio = epoch_ratio_sum / epoch_ratio_steps
        else:
            avg_ratio = float("nan")
            epoch_ratio_min = float("nan")
            epoch_ratio_max = float("nan")
        
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch + 1)
        writer.add_scalar('Loss/recon_epoch', avg_recon_loss, epoch + 1)
        writer.add_scalar('Loss/gene_epoch', avg_gene_loss, epoch + 1)
        writer.add_scalar('Loss_Unweighted/recon_epoch', avg_recon_unweighted, epoch + 1)
        writer.add_scalar('Loss_Unweighted/gene_epoch', avg_gene_unweighted, epoch + 1)
        writer.add_scalar('GradNorm/encoder', avg_gnorm_enc, epoch + 1)
        writer.add_scalar('GradNorm/cnf', avg_gnorm_cnf, epoch + 1)
        writer.add_scalar('Teacher/ratio_epoch', avg_ratio, epoch + 1)
        
        # Update epoch progress bar with average loss and grad norms
        epoch_pbar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'r_uw': f"{avg_recon_unweighted:.4f}",
            'g_uw': f"{avg_gene_unweighted:.4f}",
            'ratio': f"{avg_ratio:.4f}" if epoch_ratio_steps > 0 else 'N/A',
            'g_enc': f"{avg_gnorm_enc:.2f}",
            'g_cnf': f"{avg_gnorm_cnf:.2f}"
        })
        print(
            f"\nEpoch {epoch+1}/{EPOCHS} "
            f"Avg Loss: {avg_loss:.6f} | Recon(W/UW): {avg_recon_loss:.6f}/{avg_recon_unweighted:.6f} | Gene(W/UW): {avg_gene_loss:.6f}/{avg_gene_unweighted:.6f} "
            f"| Ratio(mean/min/max): {avg_ratio:.6f}/{epoch_ratio_min:.6f}/{epoch_ratio_max:.6f} "
            f"| GradNorm Enc: {avg_gnorm_enc:.4f} | GradNorm CNF: {avg_gnorm_cnf:.4f}"
        )
        
        # Display the binned t losses and log to tensorboard
        t_loss_strs = []
        t_loss_uw_strs = []
        t_count_strs = []
        for k in ['1.0', '0.75', '0.5', '0.25', '0.0']:
            avg_t = loss_t_sums[k] / max(loss_t_counts[k], 1)
            avg_t_uw = loss_t_unweight_sums[k] / max(loss_t_counts[k], 1)
            t_loss_strs.append(f"t_{k}: {avg_t:.6f}")
            t_loss_uw_strs.append(f"t_{k}_uw: {avg_t_uw:.6f}")
            t_count_strs.append(f"t_{k}_count: {loss_t_counts[k]}")
            writer.add_scalar(f'Loss_T/t_{k}', avg_t, epoch + 1)
            writer.add_scalar(f'Loss_T_Unweighted/t_{k}', avg_t_uw, epoch + 1)
            writer.add_scalar(f'Count_T/t_{k}', loss_t_counts[k], epoch + 1)
            
        print("Losses : " + "  |  ".join(t_loss_strs))
        print("Losses(UW): " + "  |  ".join(t_loss_uw_strs))
        print("Counts : " + "  |  ".join(t_count_strs))
        print("-" * 60)
        
        # Step schedulers
        scheduler_encoder.step()
        scheduler_cnf.step()
        writer.add_scalar('LR/encoder', optimizer_encoder.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('LR/cnf', optimizer_cnf.param_groups[0]['lr'], epoch + 1)
        
        # Save epoch checkpoint
        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
            epoch_ckpt = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'encoder_state_dict': _unwrap_state_dict(encoder),
                'encoder_ema_state_dict': encoder_ema.state_dict(),
                'cnf_state_dict': _unwrap_state_dict(cnf),
                'cnf_ema_state_dict': cnf_ema.state_dict(),
                'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                'optimizer_cnf_state_dict': optimizer_cnf.state_dict(),
                'scheduler_encoder_state_dict': scheduler_encoder.state_dict(),
                'scheduler_cnf_state_dict': scheduler_cnf.state_dict(),
            }, epoch_ckpt)
    print(f"Training Complete. Saving final checkpoint to {SAVE_PATH}...")
    final_ckpt = os.path.join(SAVE_PATH, "checkpoint_final.pt")
    torch.save({
        'epoch': EPOCHS,
        'global_step': global_step,
        'encoder_state_dict': _unwrap_state_dict(encoder),
        'encoder_ema_state_dict': encoder_ema.state_dict(),
        'cnf_state_dict': _unwrap_state_dict(cnf),
        'cnf_ema_state_dict': cnf_ema.state_dict(),
        'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
        'optimizer_cnf_state_dict': optimizer_cnf.state_dict(),
        'scheduler_encoder_state_dict': scheduler_encoder.state_dict(),
        'scheduler_cnf_state_dict': scheduler_cnf.state_dict(),
    }, final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}")
    writer.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.train <config.yml>")
        sys.exit(1)
    
    display_current_data_time()
    print("Loading configuration and starting training...")
    hp = ri.basic_input(sys.argv[-1])
    train(hp)
