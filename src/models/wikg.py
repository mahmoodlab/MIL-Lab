import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from dataclasses import dataclass

from src.models.layers import create_mlp
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GlobalAttention as GeoGlobalAttention
from src.models.mil_template import MIL
from src.builder_utils import build_model_with_cfg, _cfg
#from src.pretrained_config import PretrainedConfig

from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

_model_default_cfgs = {
    'default': _cfg(),
}

# Helper function to get activation, assuming it exists in the new codebase
# If not, you might need to define it.
def get_act(act: str):
    if act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    # Add other activations as needed
    else:
        raise NotImplementedError

# --- 1. Configuration Class ---
@dataclass
class WIKGConfig(PretrainedConfig):
    model_type = 'wikg'
    
    def __init__(self,
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_classes: int = 2,
                 topk: int = 4,
                 agg_type: str = 'bi-interaction',
                 pool: str = 'attn',
                 dropout: float = 0.25,
                 act: str = 'leaky_relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.topk = topk
        self.agg_type = agg_type
        self.pool = pool
        self.dropout = dropout
        self.act = act

# --- 2. Core WIKG-MIL Model ---
class WIKGMIL(MIL):
    def __init__(self, in_dim: int = 1024, embed_dim: int = 512, num_classes: int = 2, agg_type: str = 'bi-interaction', pool: str = 'attn', dropout: float = 0.25, act: str = 'leaky_relu', topk: int = 6, **kwargs):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.agg_type = agg_type
        self.topk = topk
        self.pool = pool
        self.dropout = dropout
        self.act = act
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.agg_type = agg_type
        self.pool = pool
        self.dropout = dropout
        self.act = act
        for k, v in kwargs.items():
            setattr(self, k, v)

        dim_hidden = embed_dim

        # Renamed '_fc1' to 'patch_embed' for consistency
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[],
            dropout=dropout,
            out_dim=dim_hidden,
            end_with_fc=False
        )

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        # Attention mechanism layers

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)
        self.scale = dim_hidden ** -0.5

        # Aggregation layers
        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError(f"Aggregation type '{agg_type}' not supported.")

        self.activation = get_act(act)
        if self.dropout > 0:
            self.message_dropout = nn.Dropout(dropout)
        else:
            self.message_dropout = nn.Identity()

        # Pooling/Readout mechanism
        if self.pool == "mean":
            self.readout = global_mean_pool
        elif self.pool == "max":
            self.readout = global_max_pool
        elif self.pool == "attn":
            attn_net = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden // 2),
                nn.LeakyReLU(),
                nn.Linear(dim_hidden // 2, 1)
            )
            self.readout = GeoGlobalAttention(attn_net)
        else:
            raise NotImplementedError(f"Pooling type '{self.pool}' not supported.")
        
        self.norm = nn.LayerNorm(dim_hidden)
        self.classifier = nn.Linear(dim_hidden, self.num_classes)

        self.initialize_weights()

    def forward_attention(self, h, attn_only=False, **kwargs):
        h = self.patch_embed(h)
        h = (h + h.mean(dim=1, keepdim=True)) * 0.5
        
        e_h = self.W_head(h)
        e_t = self.W_tail(h)

        attn_logit = (e_h @ e_t.transpose(-2, -1)) * self.scale
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # Create a full attention matrix for visualization/logging
        with torch.no_grad():
            full_attn = torch.full_like(attn_logit, float('-inf'))
            full_attn.scatter_(dim=-1, index=topk_index, src=topk_weight)
            full_attn = F.softmax(full_attn, dim=-1)

        batch_indices = torch.arange(e_t.size(0), device=h.device).view(-1, 1, 1)
        Nb_h = e_t[batch_indices, topk_index]

        topk_prob = F.softmax(topk_weight, dim=-1)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h)

        gate = torch.tanh(e_h.unsqueeze(2).expand_as(Nb_h) + eh_r)
        ka_weight = torch.einsum('bnik,bnik->bni', gate, Nb_h) # Element-wise product and sum
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(-1)
        e_Nh = torch.sum(ka_prob * Nb_h, dim=2)

        if self.agg_type == 'gcn':
            embedding = self.activation(self.linear(e_h + e_Nh))
        elif self.agg_type == 'sage':
            embedding = self.activation(self.linear(torch.cat([e_h, e_Nh], dim=-1)))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding
        
        if attn_only:
            return full_attn
        
        return embedding, full_attn

    def forward_features(self, h: torch.Tensor, attn_mask=None) -> torch.Tensor:
        h, A_raw = self.forward_attention(h, attn_only=False)
        h = self.message_dropout(h)
        
        # Squeeze batch dimension for torch_geometric pooling functions
        h_pool = self.readout(h.squeeze(0), batch=None)
        
        h_norm = self.norm(h_pool)
        return h_norm, {'attention': A_raw}

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention=False,
                return_slide_feats=False,
                ) -> torch.Tensor:
        h, log_dict = self.forward_features(h, attn_mask=attn_mask)
        logits = self.forward_head(h)
        
        cls_loss = self.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1

        if not return_attention:
            del log_dict['attention']
        if return_slide_feats:
            log_dict['slide_feats'] = h
        
        return results_dict, log_dict


# --- 3. PreTrainedModel Wrapper ---
class WIKGMILModel(PreTrainedModel):
    config_class = WIKGConfig

    def __init__(self, config: WIKGConfig, **kwargs):
        self.config = config
        super().__init__(config)
        for k, v in kwargs.items():
            setattr(config, k, v)
        
        self.model = WIKGMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            agg_type=config.agg_type,
            pool=config.pool,
            dropout=config.dropout,
            act=config.act,
            topk=config.topk,
        )
        
        # Delegate forward methods to the internal model
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


# --- 4. Register the model with AutoModel and AutoConfig ---
AutoConfig.register(WIKGConfig.model_type, WIKGConfig)
AutoModel.register(WIKGConfig, WIKGMILModel)
