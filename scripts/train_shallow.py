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
from src.dataset import TrajectoryChunkDataset, H5DirectoryChunkDataset, ShallowWaterChunkDataset
from src.models import HyperNetwork, HyperNetwork_FA, HyperNetwork_AP, HyperNetwork_ST, HyperNetwork_Perceiver, CNFRenderer, GaborRenderer
from src.models_v2 import GaborRenderer_v2, GaborRenderer_v3, HyperNetwork_Perceiver_v2, HyperNetwork_Perceiver_v3, GaborRenderer_v4, HyperNetwork_Perceiver_v4,HyperNetwork_Perceiver_v5, GaborRenderer_v5
from src.normalize import Normalizer_ts, compute_dataset_statistics


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
    target_ema_model = ema_model.module if isinstance(ema_model, nn.DataParallel) else ema_model
    with torch.no_grad():
        for k, ema_v in target_ema_model.state_dict().items():
            online_v = online_state[k].detach()
            if torch.is_floating_point(ema_v):
                ema_v.mul_(decay).add_(online_v, alpha=(1.0 - decay))
            else:
                ema_v.copy_(online_v)

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
    T_SAMPLING = getattr(hp, "t_sampling", "uniform")
    T_BETA_ALPHA = getattr(hp, "t_beta_alpha", 1.0)
    T_BETA_BETA = getattr(hp, "t_beta_beta", 3.0)
    LR_enc = getattr(hp, "lr_enc", 1e-4)
    LR_cnf = getattr(hp, "lr_cnf", 1e-4)
    STRIDE = getattr(hp, "stride", T_CHUNK)
    SAVE_PATH = getattr(hp, "save_path", "saved_models")
    USE_MULTI_GPU = getattr(hp, "use_multi_gpu", True)
    GPU_IDS_CFG = getattr(hp, "gpu_ids", None)
    SAVE_EVERY_EPOCHS = getattr(hp, "save_every_epochs", 10)
    SAVE_EVERY_STEPS = getattr(hp, "save_every_steps", 0)  # 0 to disable step-level saving
    DISTILL_LAMBDA = getattr(hp, "distill_lambda", 1.0)
    FREQ_LOSS_LAMBDA = getattr(hp, "freq_loss_lambda", 1.0)
    EMA_DECAY = getattr(hp, "ema_decay", 0.999)
    TEACHER_T_DELTA = getattr(hp, "teacher_t_delta", 0.1)
    AUTO_RESUME = getattr(hp, "auto_resume", True)
    RESUME_CHECKPOINT = getattr(hp, "resume_checkpoint", "latest")
    USE_NODE_TYPE = getattr(hp, "use_node_type", False)
    
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
        dataset = ShallowWaterChunkDataset(
            dataset_path=DATASET_PATH,
            chunk_size=T_CHUNK,
            stride=STRIDE,
            mode='train'
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=getattr(hp, "num_workers", 4))
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
    elif ENCODER_TYPE == "HyperNetwork_Perceiver_v5":
        print("Using Perceiver_v5-based HyperNetwork")
        encoder = HyperNetwork_Perceiver_v5(
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
        # HyperNetwork encodes noisy trajectories -> Z1 | Renderer decodes Z1 + Coords -> predicted trajectories
        encoder = HyperNetwork(
            t_chunk=T_CHUNK,
            channel_in=C_OUT,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH_ENC,
            num_tokens=NUM_TOKENS,
        ).to(device)

    # EMA teacher keeps a historical weighted average of the online encoder.
    encoder_ema = copy.deepcopy(encoder).to(device)
    encoder_ema.eval()
    for p in encoder_ema.parameters():
        p.requires_grad_(False)
    
    if RENDERER_TYPE == "GaborRenderer":
        print("Using MFN-based GaborRenderer")
        cnf = GaborRenderer(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v2":
        print("Using V2-based GaborRenderer")
        cnf = GaborRenderer_v2(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v3":
        print("Using V3-based GaborRenderer")
        cnf = GaborRenderer_v3(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v4":
        print("Using V4-based GaborRenderer")
        cnf = GaborRenderer_v4(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF, use_node_type=USE_NODE_TYPE).to(device)
    elif RENDERER_TYPE == "GaborRenderer_v5":
        print("Using V5-based GaborRenderer")
        cnf = GaborRenderer_v5(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF, use_node_type=USE_NODE_TYPE).to(device)
    else:
        print("Using standard CNFRenderer")
        cnf = CNFRenderer(latent_dim=LATENT_DIM, coord_dim=2, t_chunk=T_CHUNK, channel_out=C_OUT, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS_CNF).to(device)

    # Optional multi-GPU support via DataParallel
    if USE_MULTI_GPU and device.type == "cuda":
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            device_ids = _parse_gpu_ids(GPU_IDS_CFG, n_gpu)
            if len(device_ids) > 1:
                encoder = nn.DataParallel(encoder, device_ids=device_ids)
                cnf = nn.DataParallel(cnf, device_ids=device_ids)
                encoder_ema = nn.DataParallel(encoder_ema, device_ids=device_ids)
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
            _load_model_state_dict(encoder_ema, ckpt['encoder_ema_state_dict'])
        else:
            _load_model_state_dict(encoder_ema, _unwrap_state_dict(encoder))
            print("Checkpoint has no encoder_ema_state_dict, EMA initialized from online encoder.")
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

        dataset.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_gnorm_enc = 0.0
        epoch_gnorm_cnf = 0.0
        
        loss_t_sums = { '1.0': 0.0, '0.75': 0.0, '0.5': 0.0, '0.25': 0.0, '0.0': 0.0 }
        loss_t_counts = { '1.0': 0, '0.75': 0, '0.5': 0, '0.25': 0, '0.0': 0 }
        
        for step, batch_data in enumerate(dataloader):
            if USE_NODE_TYPE:
                coords_batch, mesh_info = batch_data
                traj_batch = mesh_info['fields']
                node_type = mesh_info['node_type'].to(device)
            else:
                coords_batch, traj_batch = batch_data
                node_type = None

            # coords_batch: [B, N, 2]
            # traj_batch:   [B, T, N, C]
            coords = coords_batch.to(device)
            x_real = traj_batch.to(device)
            B = x_real.shape[0]
            
            # ===== 0. Normalize inputs =====
            coords = coord_normalizer.normalize(coords)
            x_real = field_normalizer.normalize(x_real)

            # ===== A. Flow Matching Noise Addition (in DATA SPACE) =====
            if T_SAMPLING == "beta":
                # Sample continuous time step t ~ Beta(alpha, beta) for each item in batch
                m = torch.distributions.beta.Beta(torch.tensor([T_BETA_ALPHA]), torch.tensor([T_BETA_BETA]))
                t = m.sample((B,)).squeeze(-1).to(device)
            else:
                # Sample continuous time step t ~ U[0, 1] for each item in batch
                t = torch.rand(B, device=device)
            # Expand t to match image dims: [B, 1, 1, 1]
            t_expand = t.view(B, 1, 1, 1)
            
            # Sample Pure Noise matching physical shape
            noise = torch.randn_like(x_real)
            
            # Create Noisy Trajectory (OT-CFM Flow: x_t = (1-t)*noise + t*x_real)
            # Wait, standard convention: t=0 is pure noise, t=1 is clean data
            # Or vice versa. Let's define t=0 -> noise, t=1 -> clean.
            # So Flow:  x_t = (1-t) * noise + t * x_real
            x_noisy = (1 - t_expand) * noise + t_expand * x_real

            # Teacher sees lower-noise input by moving to a larger t.
            t_teacher = torch.clamp(t + TEACHER_T_DELTA, max=1.0)
            t_teacher_expand = t_teacher.view(B, 1, 1, 1)
            x_noisy_teacher = (1 - t_teacher_expand) * noise + t_teacher_expand * x_real
            
            # ===== B. Forward Pass (End to End) =====
            optimizer_encoder.zero_grad()
            optimizer_cnf.zero_grad()
            
            # 1. Predict clean dynamic latent Z1 from noisy data and coords 
            # Note: We pass `coords` and `t` so the network knows where the sensors are and noise-level
            if USE_NODE_TYPE:
                z1_pred = encoder(x_noisy, coords, t, node_type) # [B, LATENT_DIM]
            else:
                z1_pred = encoder(x_noisy, coords, t) # [B, LATENT_DIM]

            with torch.no_grad():
                if USE_NODE_TYPE:
                    z_target = encoder_ema(x_noisy_teacher, coords, t_teacher, node_type).detach()
                else:
                    z_target = encoder_ema(x_noisy_teacher, coords, t_teacher).detach()
            
            # 2. Render trajectory directly using Z1 and spatial coordinates
            # Target output corresponds to the Clean trajectory
            if USE_NODE_TYPE:
                x_pred = cnf(z1_pred, coords, node_type) # [B, T_CHUNK, N, C]
            else:
                x_pred = cnf(z1_pred, coords) # [B, T_CHUNK, N, C]
            # Instead of comparing Z, we directly enforce MSE on the physical fields!
            element_loss = F.mse_loss(x_pred, x_real, reduction='none')
            sample_loss = element_loss.reshape(B, -1).mean(dim=1)
            
            recon_loss = sample_loss.mean()
            
            # Temporal derivative loss
            # dx_pred = x_pred[:, 1:, :, :] - x_pred[:, :-1, :, :]
            # dx_real = x_real[:, 1:, :, :] - x_real[:, :-1, :, :]
            # deriv_loss = F.mse_loss(dx_pred, dx_real)
            
            distill_loss = F.mse_loss(z1_pred, z_target)
            
            # deriv_lambda = getattr(hp, "deriv_lambda", 1.0)
            loss = recon_loss + DISTILL_LAMBDA * distill_loss #+ deriv_lambda * deriv_loss
            
            # Backprop updates BOTH the HyperNetwork and the CNF at the same time
            loss.backward()
            
            # Compute gradient norm
            gnorm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=float('inf'))
            gnorm_cnf = torch.nn.utils.clip_grad_norm_(cnf.parameters(), max_norm=float('inf'))
            
            optimizer_encoder.step()
            optimizer_cnf.step()

            _ema_update(encoder_ema, encoder, EMA_DECAY)
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_distill_loss += distill_loss.item()
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
                    'encoder_ema_state_dict': _unwrap_state_dict(encoder_ema),
                    'cnf_state_dict': _unwrap_state_dict(cnf),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'optimizer_cnf_state_dict': optimizer_cnf.state_dict(),
                    'scheduler_encoder_state_dict': scheduler_encoder.state_dict(),
                    'scheduler_cnf_state_dict': scheduler_cnf.state_dict(),
                }, step_ckpt)
            
            # Bin the sample losses based on t value
            for i in range(B):
                ti = t[i].item()
                val = sample_loss[i].item()
                if ti > 0.875:
                    loss_t_sums['1.0'] += val
                    loss_t_counts['1.0'] += 1
                elif ti > 0.625:
                    loss_t_sums['0.75'] += val
                    loss_t_counts['0.75'] += 1
                elif ti > 0.375:
                    loss_t_sums['0.5'] += val
                    loss_t_counts['0.5'] += 1
                elif ti > 0.125:
                    loss_t_sums['0.25'] += val
                    loss_t_counts['0.25'] += 1
                else:
                    loss_t_sums['0.0'] += val
                    loss_t_counts['0.0'] += 1

            # Remove inner pbar postfix update
            
        avg_loss = epoch_loss / (step + 1)
        avg_recon_loss = epoch_recon_loss / (step + 1)
        avg_distill_loss = epoch_distill_loss / (step + 1)
        avg_gnorm_enc = epoch_gnorm_enc / (step + 1)
        avg_gnorm_cnf = epoch_gnorm_cnf / (step + 1)
        
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch + 1)
        writer.add_scalar('Loss/recon_epoch', avg_recon_loss, epoch + 1)
        writer.add_scalar('Loss/distill_epoch', avg_distill_loss, epoch + 1)
        writer.add_scalar('GradNorm/encoder', avg_gnorm_enc, epoch + 1)
        writer.add_scalar('GradNorm/cnf', avg_gnorm_cnf, epoch + 1)
        
        # Update epoch progress bar with average loss and grad norms
        epoch_pbar.set_postfix({
            'avg_loss': f"{avg_loss:.4f}",
            'recon': f"{avg_recon_loss:.4f}",
            'distill': f"{avg_distill_loss:.4f}",
            'g_enc': f"{avg_gnorm_enc:.2f}",
            'g_cnf': f"{avg_gnorm_cnf:.2f}"
        })
        print(
            f"\nEpoch {epoch+1}/{EPOCHS} "
            f"Average Loss: {avg_loss:.6f} | Recon: {avg_recon_loss:.6f} | Distill: {avg_distill_loss:.6f} "
            f"| GradNorm Enc: {avg_gnorm_enc:.4f} | GradNorm CNF: {avg_gnorm_cnf:.4f}"
        )
        
        # Display the binned t losses and log to tensorboard
        t_loss_strs = []
        t_count_strs = []
        for k in ['1.0', '0.75', '0.5', '0.25', '0.0']:
            avg_t = loss_t_sums[k] / max(loss_t_counts[k], 1)
            t_loss_strs.append(f"t_{k}: {avg_t:.6f}")
            t_count_strs.append(f"t_{k}_count: {loss_t_counts[k]}")
            writer.add_scalar(f'Loss_T/t_{k}', avg_t, epoch + 1)
            writer.add_scalar(f'Count_T/t_{k}', loss_t_counts[k], epoch + 1)
            
        print("Losses : " + "  |  ".join(t_loss_strs))
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
                'encoder_ema_state_dict': _unwrap_state_dict(encoder_ema),
                'cnf_state_dict': _unwrap_state_dict(cnf),
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
        'encoder_ema_state_dict': _unwrap_state_dict(encoder_ema),
        'cnf_state_dict': _unwrap_state_dict(cnf),
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
        
    hp = ri.basic_input(sys.argv[-1])
    train(hp)
