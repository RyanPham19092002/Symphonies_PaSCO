import copy
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import warnings
import MinkowskiEngine as ME

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = ME.MinkowskiLinear(d_model, n_heads  * n_points * 3)
        self.attention_weights = ME.MinkowskiLinear(d_model, n_heads  * n_points)
        self.value_proj = ME.MinkowskiLinear(d_model, d_model)
        self.output_proj = ME.MinkowskiLinear(d_model, d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     constant_(self.sampling_offsets.weight.data, 0.)
    #     thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
    #     grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    #     grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
    #     for i in range(self.n_points):
    #         grid_init[:, :, i, :] *= i + 1
    #     with torch.no_grad():
    #         self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
    #     constant_(self.attention_weights.weight.data, 0.)
    #     constant_(self.attention_weights.bias.data, 0.)
    #     xavier_uniform_(self.value_proj.weight.data)
    #     constant_(self.value_proj.bias.data, 0.)
    #     xavier_uniform_(self.output_proj.weight.data)
    #     constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, value_fea):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        Len_q, C = query.F.shape
        Len_in, C = value_fea.F.shape
        reference_points = query.C[:, 1:]
        min_coord = reference_points.min(dim=0)[0]  # (3,)
        max_coord = reference_points.max(dim=0)[0]  # (3,)
        range_coord = (max_coord - min_coord)
        scale_factor = 2

        range_voxel = range_coord // 8 + 1           #voxel size = 8x8x8
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(value_fea)      #shape [N , dim]
        value = value.view(Len_in, self.n_heads, self.d_model // self.n_heads)       #(N, head, dim/head)
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        # value = value.view(Len_in, self.n_heads, self.d_model // self.n_heads)
        # test = self.sampling_offsets(query)
        # test_F = self.sampling_offsets(query.F).view(Len_q, self.n_heads, self.n_points, 3)
        sampling_offsets = self.sampling_offsets(query).F.view(Len_q, self.n_heads, self.n_points, 3)
        sampling_offsets = (sampling_offsets * range_coord / scale_factor).int() 
        sampling_locations = reference_points[:, None, None, :] + (sampling_offsets // 8) * 8
        attention_weights = self.attention_weights(query).F.view(Len_q, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(Len_q, self.n_heads, self.n_points)
        

        indices = (sampling_locations - min_coord) / 8      #N, h, voxels, 3 
          
        indices_value = (reference_points - min_coord) / 8 
        expanded_indices_value = indices_value.unsqueeze(1).unsqueeze(2)

        mask = (indices == expanded_indices_value).all(dim=-1, keepdim=True) #N, h, voxels, 1
        index = (indices[...,0] * range_voxel[1] * range_voxel[0] + indices[...,1] * range_voxel[0]  + indices[...,2]).unsqueeze(-1)   #Lq, h, n_points,1
        
        index_flat = index.view(-1)    # [N x heads x voxels,]
        mask_flat = mask.view(-1)      # [N x heads x voxels,]

        value_fea_flat = torch.zeros((Len_in * self.n_heads * self.n_points, self.d_model // self.n_heads), device=value.F.device)      #(N * head * voxels, dim/head)
        # value_fea_flat = torch.zeros((Len_in , self.d_model), device=value.F.device) 
        safe_index_flat = index_flat.clone()
        safe_index_flat[~mask_flat] = 0         #[N x head x voxels,]
        features = value.F[safe_index_flat]  # shape [N x head x voxels, dim]
        value_fea_flat[mask_flat] = features[mask_flat] # shape [N x head x voxels, dim]
        value_fea = value_fea_flat.view(Len_in , self.n_heads , self.n_points, self.d_model // self.n_heads).permute(1,3,0,2)  # (head, dim // n_points, Lin, n_points)
        #Aggegate features by attention weighhts:
        attention_weights = attention_weights.transpose(0, 1).unsqueeze(1)  # (head, 1, Lq, n_points)
        output = (value_fea * attention_weights).sum(dim=-1)  # (head, dim, Lq)
        output = self.output_proj(output)
        output = None
        return output