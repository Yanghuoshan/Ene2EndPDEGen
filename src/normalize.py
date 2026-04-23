import numpy as np
import torch

class Normalizer_ts:
    def __init__(self, params=None, method='-11', dim=None):
        self.params = params if params is not None else []
        self.method = method
        self.dim = dim

    def fit_normalize(self, data):
        assert isinstance(data, torch.Tensor)
        if len(self.params) == 0:
            if self.method in ['-11', '01']:
                if self.dim is None:
                    self.params = (torch.max(data), torch.min(data))
                else:
                    self.params = (torch.max(data, dim=self.dim, keepdim=True)[0], torch.min(data, dim=self.dim, keepdim=True)[0])
            elif self.method == 'ms':
                if self.dim is None:
                    self.params = (torch.mean(data), torch.std(data))
                else:
                    self.params = (torch.mean(data, dim=self.dim, keepdim=True), torch.std(data, dim=self.dim, keepdim=True))
            elif self.method == 'none':
                self.params = None
        return self.normalize(data)

    def normalize(self, new_data):
        if self.method == 'none':
            return new_data
        if not hasattr(new_data, 'device'):
            return self.fnormalize(new_data, self.params, self.method)
        if getattr(new_data, 'device', 'cpu') != self.params[0].device:
            self.params = (self.params[0].to(new_data.device), self.params[1].to(new_data.device))
            
        return self.fnormalize(new_data, self.params, self.method)

    def denormalize(self, new_data_norm):
        if self.method == 'none':
            return new_data_norm
        if not hasattr(new_data_norm, 'device'):
            return self.fdenormalize(new_data_norm, self.params, self.method)
        if getattr(new_data_norm, 'device', 'cpu') != self.params[0].device:
            self.params = (self.params[0].to(new_data_norm.device), self.params[1].to(new_data_norm.device))
            
        return self.fdenormalize(new_data_norm, self.params, self.method)

    def get_params(self):
        return self.params
        
    @staticmethod
    def fnormalize(data, params, method):
        if method == '-11':
            return (data - params[1].to(data.device)) / (params[0].to(data.device) - params[1].to(data.device)) * 2 - 1
        elif method == '01':
            return (data - params[1].to(data.device)) / (params[0].to(data.device) - params[1].to(data.device))
        elif method == 'ms':
            # Add small epsilon to avoid div by zero
            std = params[1].to(data.device)
            std = torch.where(std == 0, torch.ones_like(std), std)
            return (data - params[0].to(data.device)) / std
        elif method == 'none':
            return data
        
    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == '-11':
            return (data_norm + 1) / 2 * (params[0].to(data_norm.device) - params[1].to(data_norm.device)) + params[1].to(data_norm.device)
        elif method == '01':
            return data_norm * (params[0].to(data_norm.device) - params[1].to(data_norm.device)) + params[1].to(data_norm.device)
        elif method == 'ms':
            return data_norm * params[1].to(data_norm.device) + params[0].to(data_norm.device)
        elif method == 'none':
            return data_norm

def compute_dataset_statistics(dataset, coord_method='-11', field_method='-11', coord_dim=None, field_dim=None):
    from tqdm import tqdm
    print("Computing dataset statistics for normalization...")
    
    # Coordinates
    coords = dataset.coords
    coord_params = None
    if coord_method != 'none':
        if coord_method in ['-11', '01']:
            if coord_dim is None:
                coord_params = (torch.max(coords), torch.min(coords))
            else:
                coord_params = (torch.max(coords, dim=coord_dim, keepdim=True)[0], torch.min(coords, dim=coord_dim, keepdim=True)[0])
        elif coord_method == 'ms':
            if coord_dim is None:
                coord_params = (torch.mean(coords), torch.std(coords))
            else:
                coord_params = (torch.mean(coords, dim=coord_dim, keepdim=True), torch.std(coords, dim=coord_dim, keepdim=True))

    # Fields
    field_params = None
    if field_method != 'none':
        max_val, min_val = None, None
        sum_val, sumsq_val, count = None, None, 0
        
        for sim_idx in tqdm(dataset.sim_indices, desc="Scanning dataset"):
            if hasattr(dataset, '_load_all_fields'):
                fields = dataset._load_all_fields(sim_idx)
            else:
                u, v, vo = dataset._load_sim_data(sim_idx)
                if getattr(dataset, "use_vo", False) and vo is not None:
                    fields = np.stack([u, v, vo], axis=-1)
                else:
                    fields = np.stack([u, v], axis=-1)
                
            fields_tensor = torch.tensor(fields, dtype=torch.float32)
            fields_flat = fields_tensor.reshape(-1, fields_tensor.shape[-1])
            
            if field_method in ['-11', '01']:
                if field_dim is None:
                    d_max = torch.max(fields_flat)
                    d_min = torch.min(fields_flat)
                else:
                    d_max = torch.max(fields_flat, dim=field_dim, keepdim=True)[0]
                    d_min = torch.min(fields_flat, dim=field_dim, keepdim=True)[0]
                    
                if max_val is None:
                    max_val, min_val = d_max, d_min
                else:
                    max_val = torch.max(max_val, d_max)
                    min_val = torch.min(min_val, d_min)
            elif field_method == 'ms':
                if field_dim is None:
                    d_sum =  fields_flat.sum()
                    d_sumsq = (fields_flat**2).sum()
                    d_count = fields_flat.numel()
                else:
                    d_sum = fields_flat.sum(dim=field_dim, keepdim=True)
                    d_sumsq = (fields_flat**2).sum(dim=field_dim, keepdim=True)
                    d_count = fields_flat.shape[field_dim]
                    
                if sum_val is None:
                    sum_val, sumsq_val, count = d_sum, d_sumsq, d_count
                else:
                    sum_val += d_sum
                    sumsq_val += d_sumsq
                    count += d_count
                    
        if field_method in ['-11', '01']:
            field_params = (max_val, min_val)
        elif field_method == 'ms':
            mean = sum_val / count
            var = sumsq_val / count - mean**2
            std = torch.sqrt(torch.clamp(var, min=1e-12))
            field_params = (mean, std)

    return coord_params, field_params
