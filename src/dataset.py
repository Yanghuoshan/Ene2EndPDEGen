import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

try:
    from datasets import load_from_disk
except ImportError:
    pass

def _decode_binary_array(raw: bytes, shape: tuple, dtype: np.dtype) -> np.ndarray:
    array = np.frombuffer(raw, dtype=dtype)
    return array.reshape(shape)

class TrajectoryChunkDataset(IterableDataset):
    """
    Modified TrajectoryDataset that yields fixed-length time chunks (e.g., T=16).
    Yields:
        coords: [N_points, 2]
        fields_chunk: [T_chunk, N_points, C] 
    """
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 16,
        stride: int = None,
        use_vo: bool = False,
        flatten: bool = True,
        mode: str = "train",
        sim_indices = None,
        field_dtype: np.dtype = np.float32,
        coord_dtype: np.dtype = np.float64,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.ds = load_from_disk(dataset_path)
        self.chunk_size = chunk_size
        self.stride = stride if stride is not None else chunk_size // 2
        self.use_vo = use_vo
        self.flattened = True # Forcing flatten for Point-Net style processing
        self.mode = mode
        self.field_dtype = field_dtype
        self.coord_dtype = coord_dtype
        self.seed = seed
        self.is_train = mode == "train"

        if sim_indices is None:
            self.sim_indices = list(range(len(self.ds)))
        else:
            self.sim_indices = list(sim_indices)
        
        self.num_sims = len(self.sim_indices)
        sample0 = self.ds[self.sim_indices[0]]
        self.shape_t = int(sample0["shape_t"])
        self.shape_h = int(sample0["shape_h"])
        self.shape_w = int(sample0["shape_w"])
        
        # Load static coords
        x = _decode_binary_array(sample0["x"], (self.shape_h, self.shape_w), coord_dtype)
        y = _decode_binary_array(sample0["y"], (self.shape_h, self.shape_w), coord_dtype)
        coords = np.stack([x, y], axis=-1).astype(np.float32)
        coords = coords.reshape(-1, coords.shape[-1])
        self.coords = torch.tensor(coords, dtype=torch.float32)
        
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _get_worker_info(self):
        info = get_worker_info()
        return (0, 1) if info is None else (info.id, info.num_workers)

    def _load_sim_data(self, sim_idx: int):
        record = self.ds[sim_idx]
        u = _decode_binary_array(record["u"], (self.shape_t, self.shape_h, self.shape_w), self.field_dtype)
        v = _decode_binary_array(record["v"], (self.shape_t, self.shape_h, self.shape_w), self.field_dtype)
        vo = None
        if self.use_vo and "vo" in record and record["vo"] is not None:
            vo = _decode_binary_array(record["vo"], (self.shape_t, self.shape_h, self.shape_w), self.field_dtype)
        return u, v, vo

    def __iter__(self):
        worker_id, num_workers = self._get_worker_info()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        # Simple assignment
        all_indices = np.array(self.sim_indices)
        if global_num_workers <= 1:
            worker_indices = all_indices.tolist()
        else:
            worker_indices = np.array_split(all_indices, global_num_workers)[global_worker_id].tolist()
        
        rng = np.random.default_rng(self.seed + global_worker_id + self.epoch)
        if self.is_train:
            rng.shuffle(worker_indices)
        
        for sim_idx in worker_indices:
            u, v, vo = self._load_sim_data(sim_idx)
            if self.use_vo and vo is not None:
                fields_sim = np.stack([u, v, vo], axis=-1)
            else:
                fields_sim = np.stack([u, v], axis=-1)
                
            fields_sim = fields_sim.reshape(self.shape_t, -1, fields_sim.shape[-1])
            
            # Split into chunks of size `chunk_size` using a sliding window for temporal overlap
            # If stride < chunk_size, it will create overlapping trajectories 
            # e.g., if stride=chunk_size//2, frames [0~15] and [8~23] are yielded.
            
            for t_start in range(0, self.shape_t - self.chunk_size + 1, self.stride):
                chunk = fields_sim[t_start : t_start + self.chunk_size]
                
                # Yield: coords -> [N, 2], chunk -> [T, N, C]
                yield self.coords, torch.tensor(chunk, dtype=torch.float32)

import os
import glob
import h5py

class H5DirectoryChunkDataset(IterableDataset):
    """
    Reads h5 files (1.h5, 2.h5, ...) from a directory.
    Each h5 file is treated as one full trajectory and chunked along its time axis.
    """
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 16,
        stride: int = None,
        mode: str = "train",
        seed: int = 42,
        return_mesh_info: bool = False,
        include_pressure: bool = False,
        enforce_same_trajectory_batch: bool = False,
        trajectory_batch_size: int = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.chunk_size = chunk_size
        self.stride = stride if stride is not None else chunk_size // 2
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")
        self.mode = mode
        self.seed = seed
        self.is_train = mode == "train"
        self.return_mesh_info = return_mesh_info
        self.include_pressure = include_pressure
        self.enforce_same_trajectory_batch = enforce_same_trajectory_batch
        self.trajectory_batch_size = trajectory_batch_size
        # Keep this True for compatibility with compute_dataset_statistics in normalize.py.
        self.use_vo = True

        if self.enforce_same_trajectory_batch:
            if self.trajectory_batch_size is None or self.trajectory_batch_size <= 0:
                raise ValueError(
                    "trajectory_batch_size must be a positive integer when "
                    "enforce_same_trajectory_batch=True"
                )

        # List all h5 files in the directory
        self.file_paths = glob.glob(os.path.join(self.dataset_path, "*.h5"))
        # Sort by integer filename
        try:
            self.file_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            self.file_paths.sort()
        self.num_sims = len(self.file_paths)

        if self.num_sims == 0:
            raise ValueError(f"No h5 files found in {self.dataset_path}")

        # Load per-file coords. Some datasets have different meshes per trajectory.
        self.coords_per_sim = []
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                coords = np.array(f['mesh_pos'], dtype=np.float32)
                if coords.ndim == 3 and coords.shape[-1] == 2:
                    # mesh_pos may be stored as [T, N, 2] even when mesh is time-invariant.
                    coords = coords[0] # [N, 2]
                if coords.ndim != 2 or coords.shape[-1] != 2:
                    raise ValueError(
                        f"mesh_pos must have shape [N, 2] or [T, N, 2], got {coords.shape} in {file_path}"
                    )
                self.coords_per_sim.append(torch.tensor(coords, dtype=torch.float32))

        # Keep legacy attribute used by normalize.py; concat gives global coord stats.
        self.coords = torch.cat(self.coords_per_sim, dim=0) # 

        with h5py.File(self.file_paths[0], 'r') as f:
            u0, v0, p0 = self._extract_uvp(f)
            self.traj_len = int(u0.shape[0])
            self.shape_t = self.traj_len

        # Explicit legacy-compatible attributes for normalize.py
        self.sim_indices = list(range(self.num_sims))
        self.num_sims = len(self.sim_indices)
        
        self.epoch = 0

    def _extract_uvp(self, h5_file):
        velocity = np.array(h5_file['velocity'], dtype=np.float32)
        if self.include_pressure:
            pressure = np.array(h5_file['pressure'], dtype=np.float32)

        if velocity.ndim == 2 and velocity.shape[-1] == 2:
            velocity = velocity[None, ...]
        if velocity.ndim != 3 or velocity.shape[-1] != 2:
            raise ValueError(f"velocity must have shape [T, N, 2] or [N, 2], got {velocity.shape}")

        if self.include_pressure:
            if pressure.ndim == 1:
                pressure = pressure[None, :]
            elif pressure.ndim == 3 and pressure.shape[-1] == 1:
                pressure = pressure[..., 0]
            elif pressure.ndim == 2:
                pass
            else:
                raise ValueError(f"pressure must have shape [T, N], [T, N, 1], [N] or [N, 1], got {pressure.shape}")

            if pressure.shape[0] != velocity.shape[0]:
                if pressure.shape[0] == 1 and velocity.shape[0] > 1:
                    pressure = np.repeat(pressure, velocity.shape[0], axis=0)
                else:
                    raise ValueError(
                        f"time length mismatch between velocity and pressure: "
                        f"{velocity.shape[0]} vs {pressure.shape[0]}"
                    )

            if pressure.shape[1] != velocity.shape[1]:
                raise ValueError(
                    f"node count mismatch between velocity and pressure: "
                    f"{velocity.shape[1]} vs {pressure.shape[1]}"
                )
            p = pressure
        else:
            p = None

        u = velocity[..., 0]
        v = velocity[..., 1]
        return u, v, p

    def _load_sim_data(self, sim_idx: int):
        file_path = self.file_paths[sim_idx]
        with h5py.File(file_path, 'r') as f:
            return self._extract_uvp(f)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _get_worker_info(self):
        info = get_worker_info()
        return (0, 1) if info is None else (info.id, info.num_workers)

    def __iter__(self):
        worker_id, num_workers = self._get_worker_info()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        all_indices = np.array(self.sim_indices)
        if global_num_workers <= 1:
            worker_indices = all_indices.tolist()
        else:
            worker_indices = np.array_split(all_indices, global_num_workers)[global_worker_id].tolist()
        
        rng = np.random.default_rng(self.seed + global_worker_id + self.epoch)
        if self.is_train:
            rng.shuffle(worker_indices)
            
        for sim_idx in worker_indices:
            file_path = self.file_paths[sim_idx]
            coords_sim = self.coords_per_sim[sim_idx] # [N, 2]
            with h5py.File(file_path, 'r') as f:
                u, v, p = self._extract_uvp(f)
                if self.include_pressure:
                    fields_sim = np.stack([u, v, p], axis=-1)  # [T, N, 3]
                else:
                    fields_sim = np.stack([u, v], axis=-1)  # [T, N, 2]

                if self.return_mesh_info:
                    cells = np.array(f['cells'], dtype=np.int32)
                    if cells.ndim == 3:
                        cells = cells[0]  # [N, num_vertex]
                        
                    node_type = np.array(f['node_type'], dtype=np.int32)
                    
                    if node_type.ndim == 3:
                        node_type = node_type[0]
                    elif node_type.ndim == 2 and node_type.shape[0] == fields_sim.shape[0]:
                        node_type = node_type[0] # [N] or [N, 1]
                        
                    if node_type.ndim == 1:
                        node_type = node_type[:, None]  # [N, 1]

            if fields_sim.shape[0] < self.chunk_size:
                continue

            t_starts = list(range(0, fields_sim.shape[0] - self.chunk_size + 1, self.stride))

            if self.enforce_same_trajectory_batch:
                usable = (len(t_starts) // self.trajectory_batch_size) * self.trajectory_batch_size
                if usable == 0:
                    continue
                t_starts = t_starts[:usable]

            for t_start in t_starts:
                chunk_fields = fields_sim[t_start : t_start + self.chunk_size]
                if self.return_mesh_info:
                    yield coords_sim, {
                        'fields': torch.tensor(chunk_fields, dtype=torch.float32),
                        'cells': torch.tensor(cells, dtype=torch.long),
                        'node_type': torch.tensor(node_type, dtype=torch.long)
                    }
                else:
                    yield coords_sim, torch.tensor(chunk_fields, dtype=torch.float32)

class ShallowWaterChunkDataset(IterableDataset):
    """
    Reads shallow water h5 files (traj_0000.h5, ...) from a directory.
    Each h5 file is treated as one full trajectory and chunked along its time axis.
    """
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 16,
        stride: int = None,
        mode: str = "train",
        seed: int = 42,
        sim_indices = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.chunk_size = chunk_size
        self.stride = stride if stride is not None else chunk_size // 2
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")
        self.mode = mode
        self.seed = seed
        self.is_train = mode == "train"

        # List all h5 files in the directory
        all_file_paths = glob.glob(os.path.join(self.dataset_path, "*.h5"))
        all_file_paths.sort()
        total_sims = len(all_file_paths)

        if total_sims == 0:
            raise ValueError(f"No h5 files found in {self.dataset_path}")

        if sim_indices is None:
            selected_indices = list(range(total_sims))
        else:
            selected_indices = [int(i) for i in sim_indices]
            selected_indices = list(dict.fromkeys(selected_indices))
            invalid = [i for i in selected_indices if i < 0 or i >= total_sims]
            if invalid:
                raise ValueError(
                    f"sim_indices contains out-of-range indices: {invalid}. "
                    f"Valid range is [0, {total_sims - 1}]"
                )
            if len(selected_indices) == 0:
                raise ValueError("sim_indices is empty after validation")

        self.selected_sim_indices = selected_indices
        self.file_paths = [all_file_paths[i] for i in selected_indices]
        self.num_sims = len(self.file_paths)

        # Extract mesh info from the first file and validate a shared trajectory length.
        traj_lengths = []
        with h5py.File(self.file_paths[0], 'r') as f:
            # Dynamically fetch phi and theta keys as they might have a hash suffix
            phi_key = [k for k in f['scales'].keys() if k.startswith('phi')][0]
            theta_key = [k for k in f['scales'].keys() if k.startswith('theta')][0]
            phi = f['scales'][phi_key][:]
            theta = f['scales'][theta_key][:]
            
            # Create a 2D grid coordinates [N_points, 2]
            Phi, Theta = np.meshgrid(phi, theta, indexing='ij')
            self.coords = torch.tensor(np.stack([Phi, Theta], axis=-1), dtype=torch.float32).reshape(-1, 2)
            
            v0 = f['tasks']['vorticity']
            traj_lengths.append(int(v0.shape[0]))

        for file_path in self.file_paths[1:]:
            with h5py.File(file_path, 'r') as f:
                traj_lengths.append(int(f['tasks']['vorticity'].shape[0]))

        unique_lengths = sorted(set(traj_lengths))
        if len(unique_lengths) != 1:
            raise ValueError(
                f"All shallow trajectories must share the same length for step estimation, got lengths: {unique_lengths}"
            )

        self.traj_len = unique_lengths[0]
        self.shape_t = self.traj_len
        self.traj_lens = traj_lengths
            
        self.sim_indices = list(range(self.num_sims))
        self.epoch = 0
        self.use_vo = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _load_sim_data(self, sim_idx: int):
        file_path = self.file_paths[sim_idx]
        with h5py.File(file_path, 'r') as f:
            vorticity = np.array(f['tasks']['vorticity'], dtype=np.float32)  # [T, N_x, N_y]
            height = np.array(f['tasks']['height'], dtype=np.float32)        # [T, N_x, N_y]
            return vorticity, height, None

    def _get_worker_info(self):
        info = get_worker_info()
        return (0, 1) if info is None else (info.id, info.num_workers)

    def __iter__(self):
        worker_id, num_workers = self._get_worker_info()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        all_indices = np.array(self.sim_indices)
        if global_num_workers <= 1:
            worker_indices = all_indices.tolist()
        else:
            worker_indices = np.array_split(all_indices, global_num_workers)[global_worker_id].tolist()
        
        rng = np.random.default_rng(self.seed + global_worker_id + self.epoch)
        if self.is_train:
            rng.shuffle(worker_indices)
            
        for sim_idx in worker_indices:
            vorticity, height, _ = self._load_sim_data(sim_idx)
            
            # Stack features -> [T, N_x, N_y, 2] -> [T, N_points, 2]
            fields_sim = np.stack([vorticity, height], axis=-1)
            fields_sim = fields_sim.reshape(fields_sim.shape[0], -1, fields_sim.shape[-1])

            if fields_sim.shape[0] < self.chunk_size:
                continue

            for t_start in range(0, fields_sim.shape[0] - self.chunk_size + 1, self.stride):
                chunk_fields = fields_sim[t_start : t_start + self.chunk_size]
                yield self.coords, torch.tensor(chunk_fields, dtype=torch.float32)


class MHDChunkDataset(IterableDataset):
    """
    Reads MHD h5 files from a directory.
    Each h5 file is treated as one full trajectory and chunked along its time axis.
    """
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 16,
        stride: int = None,
        mode: str = "train",
        seed: int = 42,
        downsample_factor: int = 1,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.chunk_size = chunk_size
        self.stride = stride if stride is not None else chunk_size // 2
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")
        self.mode = mode
        self.seed = seed
        self.is_train = mode == "train"
        self.downsample_factor = downsample_factor

        # List all hdf5/h5 files in the directory
        self.file_paths = glob.glob(os.path.join(self.dataset_path, "*.hdf5"))
        if not self.file_paths:
            self.file_paths = glob.glob(os.path.join(self.dataset_path, "*.h5"))
        self.file_paths.sort()
        self.num_sims = len(self.file_paths)

        if self.num_sims == 0:
            raise ValueError(f"No hdf5 files found in {self.dataset_path}")

        # Extract mesh info from the first file
        with h5py.File(self.file_paths[0], 'r') as f:
            shape_fields = f['t0_fields']['density'].shape
            # Assuming shape is (S, T, nx, ny, nz) where S is sample/trajectory index within file
            if len(shape_fields) == 5:
                num_samples, T, nx, ny, nz = shape_fields
            elif len(shape_fields) == 4:
                # If shape is (T, nx, ny, nz)
                T, nx, ny, nz = shape_fields
                num_samples = 1
            else:
                raise ValueError("Unexpected shape for density: " + str(shape_fields))
            
            # Assuming domain is [0, 1] for all dimensions if not provided
            x = np.linspace(0, 1, nx, endpoint=False)[::self.downsample_factor]
            y = np.linspace(0, 1, ny, endpoint=False)[::self.downsample_factor]
            z = np.linspace(0, 1, nz, endpoint=False)[::self.downsample_factor]
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            self.coords = torch.tensor(np.stack([X, Y, Z], axis=-1), dtype=torch.float32).reshape(-1, 3)
            
            self.traj_len = T
            self.shape_t = self.traj_len
            self.nx = nx
            self.ny = ny
            self.nz = nz
            
        self.sim_indices = []
        for i, p in enumerate(self.file_paths):
            with h5py.File(p, 'r') as f:
                s = f['t0_fields']['density'].shape
                num_s = s[0] if len(s) == 5 else 1
            for j in range(num_s):
                self.sim_indices.append((i, j))
        
        self.num_sims = len(self.sim_indices)
        self.epoch = 0
        self.use_vo = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _load_all_fields(self, sim_idx):
        file_idx, sample_idx = sim_idx
        density, mg, vel = self._load_sim_data(file_idx, sample_idx)
        # density is [T, N_x, N_y, N_z], mg and vel are [T, N_x, N_y, N_z, 3]
        density = density[..., np.newaxis]
        return np.concatenate([density, mg, vel], axis=-1)

    def _load_sim_data(self, file_idx: int, sample_idx: int):
        file_path = self.file_paths[file_idx]
        with h5py.File(file_path, 'r') as f:
            d = self.downsample_factor
            
            # Use h5py lazy slicing to only load the required sample and downsampled data into memory
            if len(f['t0_fields']['density'].shape) == 5: # [S, T, N_x, N_y, N_z]
                density = f['t0_fields']['density'][sample_idx, :, ::d, ::d, ::d]
                magnetic_field = f['t1_fields']['magnetic_field'][sample_idx, :, ::d, ::d, ::d]
                velocity = f['t1_fields']['velocity'][sample_idx, :, ::d, ::d, ::d]
            else:
                density = f['t0_fields']['density'][:, ::d, ::d, ::d]
                magnetic_field = f['t1_fields']['magnetic_field'][:, ::d, ::d, ::d]
                velocity = f['t1_fields']['velocity'][:, ::d, ::d, ::d]

            density = np.array(density, dtype=np.float32)
            magnetic_field = np.array(magnetic_field, dtype=np.float32)
            velocity = np.array(velocity, dtype=np.float32)
                
            return density, magnetic_field, velocity

    def _get_worker_info(self):
        info = get_worker_info()
        return (0, 1) if info is None else (info.id, info.num_workers)

    def __iter__(self):
        worker_id, num_workers = self._get_worker_info()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        all_indices = np.array(self.sim_indices)
        if global_num_workers <= 1:
            worker_indices = all_indices.tolist()
        else:
            worker_indices = np.array_split(all_indices, global_num_workers)[global_worker_id].tolist()
        
        rng = np.random.default_rng(self.seed + global_worker_id + self.epoch)
        if self.is_train:
            rng.shuffle(worker_indices)
            
        for file_idx, sample_idx in worker_indices:
            density, magnetic_field, velocity = self._load_sim_data(file_idx, sample_idx)
            
            # density is [T, Nx, Ny, Nz], make it [T, Nx, Ny, Nz, 1]
            density = density[..., np.newaxis]
            
            # Stack features -> [T, Nx, Ny, Nz, 7] -> [T, N_points, 7]
            fields_sim = np.concatenate([density, magnetic_field, velocity], axis=-1)
            fields_sim = fields_sim.reshape(fields_sim.shape[0], -1, fields_sim.shape[-1])

            if fields_sim.shape[0] < self.chunk_size:
                continue

            for t_start in range(0, fields_sim.shape[0] - self.chunk_size + 1, self.stride):
                chunk_fields = fields_sim[t_start : t_start + self.chunk_size]
                yield self.coords, torch.tensor(chunk_fields, dtype=torch.float32)


class IgnitHitDataset(IterableDataset):
    """
    Reads IgnitHit npz files from a directory.
    Each npz file contains a 'data' array with shape [timestep, channel, x, y].
    """
    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 16,
        stride: int = None,
        mode: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.chunk_size = chunk_size
        self.stride = stride if stride is not None else chunk_size // 2
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")
        self.mode = mode
        self.seed = seed
        self.is_train = mode == "train"

        # List all npz files in the directory
        self.file_paths = glob.glob(os.path.join(self.dataset_path, "*.npz"))
        self.file_paths.sort()
        self.num_sims = len(self.file_paths)

        if self.num_sims == 0:
            raise ValueError(f"No npz files found in {self.dataset_path}")

        # Extract info from the first file to build coordinates
        with np.load(self.file_paths[0]) as data_dict:
            first_data = data_dict["data"]  # expected shape: [30, 12, 128, 128]
            
        self.shape_t = first_data.shape[0]
        self.num_channels = first_data.shape[1]
        self.shape_x = first_data.shape[2]
        self.shape_y = first_data.shape[3]
        
        x = np.linspace(0, 1, self.shape_x, endpoint=False)
        y = np.linspace(0, 1, self.shape_y, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.coords = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(-1, 2)
        
        self.sim_indices = list(range(self.num_sims))
        self.epoch = 0
        self.use_vo = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _load_all_fields(self, sim_idx):
        file_path = self.file_paths[sim_idx]
        with np.load(file_path) as data_dict:
            data = data_dict["data"]  # [T, C, X, Y]
        return np.transpose(data, (0, 2, 3, 1))

    def _get_worker_info(self):
        info = get_worker_info()
        return (0, 1) if info is None else (info.id, info.num_workers)

    def __iter__(self):
        worker_id, num_workers = self._get_worker_info()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        all_indices = np.array(self.sim_indices)
        if global_num_workers <= 1:
            worker_indices = all_indices.tolist()
        else:
            worker_indices = np.array_split(all_indices, global_num_workers)[global_worker_id].tolist()
        
        rng = np.random.default_rng(self.seed + global_worker_id + self.epoch)
        if self.is_train:
            rng.shuffle(worker_indices)
            
        for sim_idx in worker_indices:
            file_path = self.file_paths[sim_idx]
            
            with np.load(file_path) as data_dict:
                data = data_dict["data"]  # expected shape: [T, C, N_x, N_y]
            
            # Change shape from [T, C, X, Y] to [T, X, Y, C] -> [T, N_points, C]
            fields_sim = np.transpose(data, (0, 2, 3, 1))
            fields_sim = fields_sim.reshape(fields_sim.shape[0], -1, fields_sim.shape[-1])

            if fields_sim.shape[0] < self.chunk_size:
                continue

            for t_start in range(0, fields_sim.shape[0] - self.chunk_size + 1, self.stride):
                chunk_fields = fields_sim[t_start : t_start + self.chunk_size]
                yield self.coords, torch.tensor(chunk_fields, dtype=torch.float32)
