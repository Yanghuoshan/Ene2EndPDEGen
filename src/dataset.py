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
