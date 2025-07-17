from dataclasses import dataclass

import torch.nn.functional as F
from torch import nn

from src.models.mil_template import MIL
from transformers import PretrainedConfig
from transformers import PreTrainedModel, AutoConfig, AutoModel

MODEL_TYPE = 'rrtmil'


class PPEG(nn.Module):
    def __init__(self, dim=512, k=7, conv_1d=False, bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                           (k, 1), 1,
                                                                                                           (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (5, 1), 1,
                                                                                                            (5 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (3, 1), 1,
                                                                                                            (3 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))

        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x = torch.cat([x, torch.zeros((B, zero_pad, C), device=x.device)], dim=1)
            add_length += zero_pad

        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        if add_length > 0:
            x = x[:, :-add_length]
        return x


class PEG(nn.Module):
    def __init__(self, dim=512, k=7, bias=True, conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                           (k, 1), 1,
                                                                                                           (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length > 0:
            x = x[:, :-add_length]

        return x


class SINCOS(nn.Module):
    def __init__(self, embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)

    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        # B, N, C = x.shape
        B, H, W, C = x.shape
        # # padding
        pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)

        x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)

        return x


class Attention(nn.Module):
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D, bias=bias)]

        if act == 'gelu':
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K, bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self, x, no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        else:
            return x, A


class AttentionGated(nn.Module):
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [
            nn.Linear(self.L, self.D, bias=bias),
        ]

        self.attention_a += [get_act(act)]

        self.attention_b = [nn.Linear(self.L, self.D, bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

    def forward(self, x, no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        else:
            return x, A


class DAttention(nn.Module):
    def __init__(self, input_dim=512, act='relu', gated=False, bias=False, dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim, act, bias, dropout)
        else:
            self.attention = Attention(input_dim, act, bias, dropout)

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None, len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _, ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, no_norm=False, mask_enable=False):

        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x, mask_ids, len_keep)

        x, attn = self.attention(x, no_norm)

        if return_attn:
            return x.squeeze(1), attn.squeeze(1)
        else:
            return x.squeeze(1)


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
                      0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, num_classes=2, mask_ratio=0., mlp_dim=512, cls_attn=True, attn_index='max'):
        super(DSMIL, self).__init__()

        self.i_classifier = nn.Sequential(
            nn.Linear(mlp_dim, num_classes))
        self.b_classifier = BClassifier(mlp_dim, num_classes)

        self.cls_attn = cls_attn
        self.attn_index = attn_index

        self.mask_ratio = mask_ratio

    def attention(self, x, no_norm=False, label=None, criterion=None, return_attn=False):
        ps = x.size(1)
        feats = x.squeeze(0)
        classes = self.i_classifier(feats)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        classes_bag, _ = torch.max(classes, 0)

        if return_attn:
            # 通过bag和inst综合判断
            if self.attn_index == 'max':
                attn, _ = torch.max(classes, -1) if self.cls_attn else torch.max(A, -1)
            elif self.attn_index == 'label':
                if label is None:
                    pred = 0.5 * torch.softmax(prediction_bag, dim=-1) + 0.5 * torch.softmax(classes_bag, dim=-1)
                    _, _attn_idx = torch.max(pred.squeeze(), 0)
                    attn = classes[:, int(_attn_idx)] if self.cls_attn else A[:, int(_attn_idx)]
                else:
                    attn = classes[:, label[0]] if self.cls_attn else A[:, label[0]]
            else:
                attn = classes[:, int(self.attn_index)] if self.cls_attn else A[:, int(self.attn_index)]
            attn = attn.unsqueeze(0)
        else:
            attn = None

        if self.training and criterion is not None:
            max_loss = criterion(classes_bag.view(1, -1), label)
            return prediction_bag, attn, B, max_loss
        else:
            return prediction_bag, attn, B, classes_bag.unsqueeze(0)

    def random_masking(self, x, mask_ratio, ids_shuffle=None, len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        N, L, D = x.shape  # batch, length, dim
        if ids_shuffle is None:
            # sort noise for each sample
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        else:
            _, ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, no_norm=False, mask_enable=False, **kwargs):

        if mask_enable and (self.mask_ratio > 0. or mask_ids is not None):
            x, _, _ = self.random_masking(x, self.mask_ratio, mask_ids, len_keep)

        _label = kwargs['label'] if 'label' in kwargs else None
        _criterion = kwargs['criterion'] if 'criterion' in kwargs else None

        prediction_bag, attn, B, other = self.attention(x, no_norm, _label, _criterion, return_attn=return_attn)

        logits = prediction_bag

        if return_attn:
            return B, logits, other, attn
        else:
            return B, logits, other


def get_act(act):
    if act.lower() == 'relu':
        return torch.nn.ReLU()
    elif act.lower() == 'gelu':
        return torch.nn.GELU()
    elif act.lower() == 'leakyrelu':
        return torch.nn.LeakyReLU()
    elif act.lower() == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act.lower() == 'tanh':
        return torch.nn.Tanh()
    elif act.lower() == 'silu':
        return torch.nn.SiLU()
    else:
        raise ValueError(f'Invalid activation function: {act}')


import torch.nn as nn
from src.components.nystrom_attention import NystromAttention


# --------------------------------------------------------
# Modified by Swin@Microsoft
# --------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions


def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class InnerAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted region.
    Args:
        dim (int): Number of input channels.
        region_size (tuple[int]): The height and width of the region.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, head_dim=None, region_size=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., conv=True, conv_k=15, conv_2d=False, conv_bias=True, conv_type='attn'):

        super().__init__()
        self.dim = dim
        self.region_size = [region_size, region_size] if region_size is not None else None  # Wh, Ww
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        if region_size is not None:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.region_size[0] - 1) * (2 * self.region_size[1] - 1),
                            num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the region
            coords_h = torch.arange(self.region_size[0])
            coords_w = torch.arange(self.region_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.region_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.region_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.region_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv_2d = conv_2d
        self.conv_type = conv_type
        if conv:
            kernel_size = conv_k
            padding = kernel_size // 2

            if conv_2d:
                if conv_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, kernel_size, padding=padding, groups=num_heads,
                                        bias=conv_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, kernel_size, padding=padding,
                                        groups=head_dim * num_heads, bias=conv_bias)
            else:
                if conv_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (kernel_size, 1), padding=(padding, 0), groups=num_heads,
                                        bias=conv_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, (kernel_size, 1),
                                        padding=(padding, 0), groups=head_dim * num_heads, bias=conv_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
            mask: (0/-inf) mask with shape of (num_regions, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and self.conv_type == 'attn':
            pe = self.pe(attn)
            attn = attn + pe

        if self.region_size is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.region_size[0] * self.region_size[1], self.region_size[0] * self.region_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.conv_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            v = v + pe.reshape(B_, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)

        if self.pe is not None and self.conv_type == 'value_af':
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            x = x + pe.reshape(B_, self.num_heads * self.head_dim, N).transpose(-1, -2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, region_size={self.region_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 region with token length of N
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class RegionAttntion(nn.Module):
    def __init__(self, dim, input_resolution=None, head_dim=None, num_heads=8, region_size=0, shift_size=False,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8, conv=False, rpe=False,
                 min_region_num=0, min_region_ratio=0.0, region_attn='native', **kawrgs):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.shift_size = shift_size
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        self.rpe = rpe

        if self.region_size is not None:
            self.region_num = None
        self.fused_region_process = False

        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim, num_heads=num_heads, region_size=self.region_size if rpe else None,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, conv=conv, **kawrgs)
        elif region_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )

        self.attn_mask = None

    def padding(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num) and not self.rpe:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # R-MSA
        attn_regions = self.attn(x_regions, mask=self.attn_mask)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, L, C = x.shape
        # padding
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % 2
        H, W = H + _n, W + _n
        add_length = H * W - L
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransLayer1(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head=8, drop_out=0.1, drop_path=0., need_down=False,
                 need_reduce=False, down_ratio=2, ffn=False, ffn_act='gelu', mlp_ratio=4., trans_dim=64, n_cycle=1,
                 attn='ntrans', n_region=8, epeg=False, shift_size=False, region_size=0, rpe=False, min_region_num=0,
                 min_region_ratio=0.0, qkv_bias=True, **kwargs):
        super().__init__()

        if need_reduce:
            self.reduction = nn.Linear(dim, dim // down_ratio, bias=False)
            dim = dim // down_ratio
        else:
            self.reduction = nn.Identity()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=trans_dim,  # dim // 8
                heads=head,
                num_landmarks=256,  # number of landmarks dim // 2
                pinv_iterations=6,
                # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual=True,
                # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out
            )
        elif attn == 'rrt':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=trans_dim,
                conv=epeg,
                shift_size=shift_size,
                region_size=region_size,
                rpe=rpe,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                **kwargs
            )
        else:
            raise NotImplementedError

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop_out) if ffn else nn.Identity()

        self.downsample = PatchMerging(None, dim) if need_down else nn.Identity()

        self.n_cycle = n_cycle

    def forward(self, x, need_attn=False):
        attn = None
        for i in range(self.n_cycle):
            x, attn = self.forward_trans(x, need_attn=need_attn)

        if need_attn:
            return x, attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None

        x = self.reduction(x)
        B, L, C = x.shape

        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x + self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.downsample(x)

        return x, attn


class RRTEncoder(nn.Module):
    def __init__(self, mlp_dim=512, pos_pos='ppeg', pos='none', peg_k=7, attn='ntrans',
                 region_num=8, drop_out=0.1, n_layers=1, n_heads=8,
                 multi_scale=False, drop_path=0.1, pool='attn', da_act='tanh',
                 reduce_ratio=0, ffn=False, ffn_act='gelu', mlp_ratio=4., da_gated=False,
                 da_bias=False, da_dropout=False, trans_dim=64, n_cycle=1, epeg=True,
                 rpe=False, region_size=0, min_region_num=0, min_region_ratio=0.0,
                 qkv_bias=True, shift_size=False, peg_bias=True, peg_1d=False, **kwargs):
        super(RRTEncoder, self).__init__()

        self.final_dim = mlp_dim // (2 ** reduce_ratio) if reduce_ratio > 0 else mlp_dim
        if multi_scale:
            self.final_dim *= (2 ** (n_layers - 1))

        self.pool = pool
        if self.pool == 'attn':
            self.pool_fn = DAttention(self.final_dim, da_act, gated=da_gated, bias=da_bias, dropout=da_dropout)

        self.norm = nn.LayerNorm(self.final_dim)
        self.layer1 = TransLayer1(dim=mlp_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path,
                                  need_down=multi_scale, need_reduce=reduce_ratio != 0,
                                  down_ratio=2 ** reduce_ratio, ffn=ffn, ffn_act=ffn_act,
                                  mlp_ratio=mlp_ratio, trans_dim=trans_dim, n_cycle=n_cycle,
                                  n_region=region_num, epeg=epeg, rpe=rpe,
                                  region_size=region_size, min_region_num=min_region_num,
                                  min_region_ratio=min_region_ratio, qkv_bias=qkv_bias,
                                  shift_size=shift_size, **kwargs)

        if n_layers >= 2:
            layers = []
            current_dim = mlp_dim // (2 ** reduce_ratio) if reduce_ratio > 0 else mlp_dim
            if multi_scale:
                current_dim *= 2

            for i in range(n_layers - 2):
                layers.append(TransLayer1(dim=current_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path,
                                          need_down=multi_scale, ffn=ffn, ffn_act=ffn_act,
                                          mlp_ratio=mlp_ratio, trans_dim=trans_dim, n_cycle=n_cycle,
                                          n_region=region_num, epeg=epeg, rpe=rpe,
                                          region_size=region_size, min_region_num=min_region_num,
                                          min_region_ratio=min_region_ratio, qkv_bias=qkv_bias))
                if multi_scale:
                    current_dim *= 2

            layers.append(TransLayer1(dim=current_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path,
                                      ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                                      trans_dim=trans_dim, n_cycle=n_cycle, n_region=region_num,
                                      epeg=epeg, rpe=rpe, region_size=region_size,
                                      min_region_num=min_region_num, min_region_ratio=min_region_ratio,
                                      qkv_bias=qkv_bias, shift_size=shift_size, **kwargs))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Identity()

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    def forward(self, x, return_attn=False, no_norm=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        x = self.layer1(x)
        x = self.layers(x)
        x = self.norm(x)

        if self.pool == 'attn':
            if return_attn:
                return self.pool_fn(x, return_attn=True, no_norm=no_norm)
            return self.pool_fn(x)

        return x.mean(dim=1)  # Fallback for other pooling types


class RRTMIL(MIL):
    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            mlp_dim: int = 512,
            dropout: float = 0.25,
            act: str = 'relu',
            num_classes: int = 2,
            n_layers: int = 4,
            n_heads: int = 8,
            drop_out: float = 0.0,
            drop_path: float = 0.0,
            multi_scale: bool = False,
            ffn: bool = True,
            ffn_act: str = 'gelu',
            mlp_ratio: float = 4.0,
            trans_dim: int = 512,
            n_cycle: int = 1,
            region_num: int = 1,
            epeg: bool = False,
            rpe: bool = False,
            region_size: int = 7,
            min_region_num: int = 1,
            min_region_ratio: float = 0.0,
            qkv_bias: bool = True,
            shift_size: bool = False,
            pos: str = 'ppeg',
            pos_pos: str = 'none',
            peg_k: int = 7,
            peg_bias: bool = True,
            peg_1d: bool = False,
            pool: str = 'attn',
            mask_ratio: float = 0.0,
            cls_attn: bool = True,
            attn_index: str = 'max',
            **kwargs
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        patch_to_emb = [nn.Linear(in_dim, embed_dim)]
        if act.lower() == 'relu':
            patch_to_emb.append(nn.ReLU())
        elif act.lower() == 'gelu':
            patch_to_emb.append(nn.GELU())

        self.patch_embed = nn.Sequential(*patch_to_emb)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.encoder = RRTEncoder(mlp_dim=mlp_dim, n_layers=n_layers, n_heads=n_heads, drop_out=drop_out,
                                  drop_path=drop_path, multi_scale=multi_scale, ffn=ffn, ffn_act=ffn_act,
                                  mlp_ratio=mlp_ratio, trans_dim=trans_dim, n_cycle=n_cycle, region_num=region_num,
                                  epeg=epeg, rpe=rpe, region_size=region_size, min_region_num=min_region_num,
                                  min_region_ratio=min_region_ratio, qkv_bias=qkv_bias, shift_size=shift_size, pos=pos,
                                  pos_pos=pos_pos, peg_k=peg_k, peg_bias=peg_bias, peg_1d=peg_1d, pool=pool,
                                  mask_ratio=mask_ratio, cls_attn=cls_attn, attn_index=attn_index, **kwargs)

        if num_classes > 0:
            self.classifier = nn.Linear(self.encoder.final_dim, num_classes)

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True):
        h = self.patch_embed(h)
        h = self.dropout(h)

        if attn_only:
            _, A = self.encoder(h, return_attn=True)
            return A

        h, A = self.encoder(h, return_attn=True)
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None):
        h, A = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        return h, {'attention': A}

    def forward_head(self, h: torch.Tensor):
        return self.classifier(h)

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> tuple[dict, dict]:

        h, log_dict = self.forward_features(h, attn_mask=attn_mask)
        logits = self.forward_head(h)

        cls_loss = self.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = h

        return results_dict, log_dict


@dataclass
class RRTMILConfig(PretrainedConfig):
    """Configuration class for RRTMIL models."""
    model_type = MODEL_TYPE

    def __init__(self,
                 in_dim: int = 1024,
                 mlp_dim: int = 512,
                 embed_dim: int = 512,
                 act: str = "relu",
                 num_classes: int = 2,
                 dropout: float = 0.25,
                 pos_pos: str = "none",
                 pos: str = "ppeg",
                 peg_k: int = 7,
                 attn: str = "ntrans",
                 pool: str = "attn",
                 region_num: int = 8,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 multi_scale: bool = False,
                 drop_path: float = 0.0,
                 da_act: str = "relu",
                 trans_dropout: float = 0.1,
                 ffn: bool = False,
                 ffn_act: str = "gelu",
                 mlp_ratio: float = 4.0,
                 da_gated: bool = False,
                 da_bias: bool = False,
                 da_dropout: bool = False,
                 trans_dim: int = 64,
                 n_cycle: int = 1,
                 epeg: bool = False,
                 min_region_num: int = 0,
                 qkv_bias: bool = True,
                 shift_size: bool = False,
                 no_norm: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.embed_dim = embed_dim
        self.act = act
        self.num_classes = num_classes
        self.num_classes = num_classes  # alias for compatibility
        self.dropout = dropout
        self.pos_pos = pos_pos
        self.pos = pos
        self.peg_k = peg_k
        self.attn = attn
        self.pool = pool
        self.region_num = region_num
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.multi_scale = multi_scale
        self.drop_path = drop_path
        self.da_act = da_act
        self.trans_dropout = trans_dropout
        self.ffn = ffn
        self.ffn_act = ffn_act
        self.mlp_ratio = mlp_ratio
        self.da_gated = da_gated
        self.da_bias = da_bias
        self.da_dropout = da_dropout
        self.trans_dim = trans_dim
        self.n_cycle = n_cycle
        self.epeg = epeg
        self.min_region_num = min_region_num
        self.qkv_bias = qkv_bias
        self.shift_size = shift_size
        self.no_norm = no_norm


class RRTMILModel(PreTrainedModel):
    """Hugging Face wrapper for the RRTMIL model."""
    config_class = RRTMILConfig

    def __init__(self, config: RRTMILConfig, **kwargs):
        self.config = config
        super().__init__(config)
        self.model = RRTMIL(
            in_dim=config.in_dim,
            mlp_dim=config.mlp_dim,
            embed_dim=config.embed_dim,
            act=config.act,
            num_classes=config.num_classes,
            dropout=config.dropout,
            pos_pos=config.pos_pos,
            pos=config.pos,
            peg_k=config.peg_k,
            attn=config.attn,
            pool=config.pool,
            region_num=config.region_num,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            multi_scale=config.multi_scale,
            drop_path=config.drop_path,
            da_act=config.da_act,
            trans_dropout=config.trans_dropout,
            ffn=config.ffn,
            ffn_act=config.ffn_act,
            mlp_ratio=config.mlp_ratio,
            da_gated=config.da_gated,
            da_bias=config.da_bias,
            da_dropout=config.da_dropout,
            trans_dim=config.trans_dim,
            n_cycle=config.n_cycle,
            epeg=config.epeg,
            min_region_num=config.min_region_num,
            qkv_bias=config.qkv_bias,
            shift_size=config.shift_size,
            no_norm=config.no_norm
        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


# Register the model with Hugging Face AutoClass
AutoConfig.register(MODEL_TYPE, RRTMILConfig)
AutoModel.register(RRTMILConfig, RRTMILModel)
