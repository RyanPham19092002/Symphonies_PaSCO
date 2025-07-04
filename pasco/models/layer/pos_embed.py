from itertools import chain

import torch
import torch.nn as nn


class LearnableSqueezePositionalEncoding(nn.Module):

    def __init__(self, num_embeds, embed_dims, squeeze_dims=None):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(n_emb * squeeze_dims[i] if squeeze_dims else 1, embed_dims // 2)
            for i, n_emb in enumerate(num_embeds)
        ])
        self.proj = nn.Linear(embed_dims // 2 * len(num_embeds), embed_dims)
        self.shape = num_embeds
        self.squeeze_dims = squeeze_dims

    def forward(self, voxel_coordinates):
        embeds = []
        for i, s in enumerate(self.shape):
            shape = [1 for _ in self.shape]
            shape[i] = s * self.squeeze_dims[i] if self.squeeze_dims else 1
            embed = self.embeds[i].weight.reshape(1, *shape, -1).expand(
                1, *[
                    s * self.squeeze_dims[i] if self.squeeze_dims else 1
                    for i, s in enumerate(self.shape)
                ], -1)
            embeds.append(embed)
        embed = torch.cat(embeds, dim=-1)
        # embed = embed[:, voxel_coordinates, :]
        if self.squeeze_dims:
            shape = list(chain(*[(s, self.squeeze_dims[i]) for i, s in enumerate(self.shape)]))
            dims = [2 * i + 1
                    for i in range(len(self.shape))] + [2 * i + 2 for i in range(len(self.shape))]
            
            bs = embed.shape[0]
            N = voxel_coordinates.shape[0]

            x_idx = voxel_coordinates[:, 0]
            y_idx = voxel_coordinates[:, 1]
            z_idx = voxel_coordinates[:, 2]

            batch_idx = torch.arange(bs, device=voxel_coordinates.device).unsqueeze(1).expand(-1, N).type(torch.long)
            x_idx_b = x_idx.unsqueeze(0).expand(bs, -1).type(torch.long)
            y_idx_b = y_idx.unsqueeze(0).expand(bs, -1).type(torch.long)
            z_idx_b = z_idx.unsqueeze(0).expand(bs, -1).type(torch.long)
            embed = embed.reshape(1, *shape, -1).permute(0, *dims, -1).to(voxel_coordinates.device)
            embed = embed[batch_idx, x_idx_b, y_idx_b, z_idx_b]
            embed = embed.flatten(1, -2).cpu()

            # embed = embed.reshape(1, *shape, -1).permute(0, *dims, -1).flatten(1, -2)             #origninal

        embed = self.proj(embed).to(voxel_coordinates.device)
        return embed


class FactorizedPositionEmbedding(nn.Module):

    def __init__(self, num_embeds, embed_dims, squeeze_dims=None):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(n_emb * squeeze_dims[i] if squeeze_dims else 1, embed_dims)
            for i, n_emb in enumerate(num_embeds)
        ])
        self.shape = num_embeds
        self.squeeze_dims = squeeze_dims

    def forward(self):
        embeds = 1
        for i, s in enumerate(self.shape):
            shape = [1 for _ in self.shape]
            shape[i] = s * self.squeeze_dims[i] if self.squeeze_dims else 1
            embed = self.embeds[i].weight.reshape(1, *shape, -1).expand(
                1, *[
                    s * self.squeeze_dims[i] if self.squeeze_dims else 1
                    for i, s in enumerate(self.shape)
                ], -1)
            embeds = embeds * embed
        if self.squeeze_dims:
            shape = list(chain(*[(s, self.squeeze_dims[i]) for i, s in enumerate(self.shape)]))
            dims = [2 * i + 1
                    for i in range(len(self.shape))] + [2 * i + 2 for i in range(len(self.shape))]
            embed = embed.reshape(1, *shape, -1).permute(0, *dims, -1).flatten(1, -2)
        return embed