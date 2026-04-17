import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math


DEFAULT_W0 = 30.

########################
# Initialization methods
########################
def __check_Linear_weight(m):
    if isinstance(m, nn.Linear):
        if hasattr(m, 'weight'):
            return True
    return False

def init_weights_normal(m):
    if __check_Linear_weight(m):
        nn.init.kaiming_normal_(
            m.weight, 
            a=0.0, 
            nonlinearity='relu', 
            mode='fan_in'
        )

def init_weights_selu(m):
    if __check_Linear_weight(m):
        num_input = m.weight.size(-1)
        nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))

def init_weights_elu(m):
    if __check_Linear_weight(m):
        num_input = m.weight.size(-1)
        nn.init.normal_(
            m.weight, 
            std=math.sqrt(1.5505188080679277) / math.sqrt(num_input)
        )

def init_weights_xavier(m):
    if __check_Linear_weight(m):
        nn.init.xavier_normal_(m.weight)

def sine_init(m, w0 = DEFAULT_W0):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor DEFAULT_W0
            m.weight.uniform_(
                -math.sqrt(6 / num_input) / w0, 
                math.sqrt(6 / num_input) / w0
            )

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor DEFAULT_W0
            m.weight.uniform_(-1 / num_input, 1 / num_input)


########################
# Activation functions 
########################
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()
    def forward(self,x):
        return x*self.Sigmoid(x)

class Sine(nn.Module):
    def __init__(self, w0 = DEFAULT_W0):
        self.w0 = w0
        super().__init__()

    def forward(self, input):
        return torch.sin(self.w0 * input)

class Sine_tw(nn.Module):
    def __init__(self, w0 = DEFAULT_W0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor([w0], dtype = torch.float32))

    def forward(self, input):
        return torch.sin(self.w0 * input)


NLS_AND_INITS = {
    'sine': (Sine(), sine_init, first_layer_sine_init), 
    'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
    'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
    'tanh': (nn.Tanh(), init_weights_xavier, None),
    'selu': (nn.SELU(inplace=True), init_weights_selu, None),
    'softplus': (nn.Softplus(), init_weights_normal, None),
    'elu': (nn.ELU(inplace=True), init_weights_elu, None),
    'swish': (Swish(), init_weights_xavier, None),
}

########################
# Basic layers 
########################
class BatchLinear(nn.Linear):
    '''
    This is a linear transformation implemented manually. It also allows maually input parameters. 
    for initialization, (in_features, out_features) needs to be provided. 
    weight is of shape (out_features*in_features)
    bias is of shape (out_features)
    '''
    __doc__ = nn.Linear.__doc__
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']
        output = torch.matmul(input, weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        if bias is not None: 
            output += bias.unsqueeze(-2)
        return output


class FeatureMapping():
    '''
    This is feature mapping class for fourier feature networks 
    '''
    def __init__(self, in_features, mode = 'basic', 
                 gaussian_mapping_size = 256, gaussian_rand_key = 0, gaussian_tau = 1.,
                 pe_num_freqs = 4, pe_scale = 2, pe_init_scale = 1, pe_use_nyquist=True, pe_lowest_dim = None, 
                 rbf_out_features = None, rbf_range = 1., rbf_std=0.5):
        self.mode = mode
        if mode == 'basic':
            self.B = np.eye(in_features)
        elif mode == 'gaussian':
            rng = np.random.default_rng(gaussian_rand_key)
            self.B = rng.normal(loc=0., scale=gaussian_tau, size=(gaussian_mapping_size, in_features))
        elif mode == 'positional':
            if pe_use_nyquist == 'True' and pe_lowest_dim:  
                pe_num_freqs = self.get_num_frequencies_nyquist(pe_lowest_dim)
            self.B = pe_init_scale * np.vstack([(pe_scale**i)* np.eye(in_features) for i in range(pe_num_freqs)])
            self.dim = self.B.shape[0]*2
        elif mode == 'rbf':
            self.centers = nn.Parameter(torch.empty((rbf_out_features, in_features), dtype = torch.float32))
            self.sigmas = nn.Parameter(torch.empty(rbf_out_features, dtype = torch.float32))
            nn.init.uniform_(self.centers, -1*rbf_range, rbf_range)
            nn.init.constant_(self.sigmas, rbf_std)

    def __call__(self, input):
        if self.mode in ['basic', 'gaussian', 'positional']: 
            return self.fourier_mapping(input, self.B)
        elif self.mode == 'rbf':
            return self.rbf_mapping(input)
            
    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    @staticmethod
    def fourier_mapping(x, B):
        if B is None:
            return x
        else:
            B = torch.tensor(B, dtype = torch.float32, device = x.device)
            x_proj = (2.*np.pi*x) @ B.T
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
    def rbf_mapping(self, x):
        size = (x.shape[:-1])+ self.centers.shape
        x = x.unsqueeze(-2).expand(size)
        distances = (x - self.centers).pow(2).sum(-1) * self.sigmas
        return self.gaussian(distances)
    
    @staticmethod
    def gaussian(alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi


########################
# SIREN Auto Decoders
########################
# siren auto decoder: FILM
class SIRENAutodecoder_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None, bias_init=None,
                 premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode == None: 
            self.premap_layer = FeatureMapping(in_coord_features, mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     

        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        self.net1 = nn.ModuleList([BatchLinear(in_coord_features, hidden_features)] + 
                                  [BatchLinear(hidden_features, hidden_features) for i in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features, out_features)])
        self.net2 = nn.ModuleList([BatchLinear(in_latent_features, hidden_features, bias = False) for i in range(num_hidden_layers+1)])

        if self.weight_init is not None:
            self.net1.apply(self.weight_init)
            self.net2.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net1[0].apply(first_layer_init)
            self.net2[0].apply(first_layer_init)
        if bias_init is not None:
            self.net2.apply(bias_init)

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>

        # premap 
        if not self.premap_mode == None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass it through the nf network 
        for i in range(len(self.net1) - 1):
            x = self.net1[i](x) + self.net2[i](latents)
            x = self.nl(x)
        x = self.net1[-1](x)
        return x 

    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False


class SIRENRenderer(nn.Module):
    """
    SIREN-based renderer adapted to support multi-latent cross-attention 
    and frequency domain output similarly to GaborRenderer_v2.
    """
    def __init__(self, latent_dim=256, coord_dim=2, t_chunk=16, channel_out=2, hidden_dim=256, num_layers=4, use_node_type=False, node_type_dim=16, encoded_coord_dim=128, nonlinearity='sine'):
        super().__init__()
        self.t_chunk = t_chunk
        self.channel_out = channel_out
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_node_type = use_node_type

        self.nl, self.nl_weight_init, self.first_layer_init = NLS_AND_INITS[nonlinearity]

        in_coord_features = coord_dim

        self.net1 = nn.ModuleList([BatchLinear(in_coord_features, hidden_dim)] + 
                                  [BatchLinear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        self.net2 = nn.ModuleList(
            [BatchLinear(latent_dim, hidden_dim, bias=False) for _ in range(num_layers + 1)]
        )

        in_dim_query = encoded_coord_dim + (node_type_dim if use_node_type else 0)
        self.query_proj = nn.Sequential(
            nn.Linear(in_dim_query, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)

        self.freq_dim = (t_chunk // 2 + 1)
        out_dim = self.freq_dim * 2 * channel_out
            
        self.final_linear = nn.Linear(hidden_dim, out_dim)

        # initialization 
        self.net1.apply(self.nl_weight_init)
        self.net2.apply(self.nl_weight_init)
        if self.first_layer_init is not None:
            self.net1[0].apply(self.first_layer_init)
            self.net2[0].apply(self.first_layer_init)

    def forward(self, z_multi, coords, coords_encoded, type_embeds=None):
        B, N, _ = coords.shape
        x0 = coords
        
        # Extract location-specific latent via Cross-Attention
        if self.use_node_type and type_embeds is not None:
            query_input = torch.cat([coords_encoded, type_embeds], dim=-1)
        else:
            query_input = coords_encoded
            
        q = self.query_proj(query_input)
        q = self.norm_q(q)
        kv = self.norm_kv(z_multi)
        z, _ = self.cross_attn(q, kv, kv)
        
        x = x0
        for i in range(len(self.net1)):
            x = self.net1[i](x) + self.net2[i](z)
            x = self.nl(x)
            
        out = self.final_linear(x)
        
        # Reshape to expected sequence shape: [B, F, N, channel_out, 2]
        out_freq_real = out.view(B, N, self.freq_dim, self.channel_out, 2)
        out_freq_real = out_freq_real.permute(0, 2, 1, 3, 4)
        out_freq_complex = torch.view_as_complex(out_freq_real) # [B, F, N, channel_out]

        # Convert frequency domain back to time domain
        out = torch.fft.irfft(out_freq_complex, n=self.t_chunk, dim=1, norm="ortho") # [B, T_chunk, N, channel_out]
        return out



