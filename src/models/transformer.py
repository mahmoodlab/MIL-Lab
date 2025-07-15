from src.models.mil_template import MIL
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig

import torch
import torch.nn as nn
from src.components import create_mlp


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8, dropout=0.1, mlp_dim=None):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                       dropout=dropout, batch_first=False)
        if mlp_dim is not None and mlp_dim > 0:
            self.mlp = create_mlp(in_dim=dim, hid_dims=[mlp_dim], dropout=dropout, out_dim=dim, end_with_fc=False)
            self.norm2 = norm_layer(dim)
        else:
            self.mlp = None
            self.norm2 = None

    def forward(self, x, need_weights=False) -> torch.Tensor:
        """
        Forward pass for the transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D).
            need_weights (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, D) if need_weights is False.
            (torch.Tensor, torch.Tensor): Output tensor and attention weights if need_weights is True.
        """
        x, was_unbatched = MIL.ensure_batched(x, return_was_unbatched=True)
        attention_weights = None
        device = next(self.norm.parameters()).device
        x = x.to(device)
        x = MIL.ensure_unbatched(x)
        x = x.view(x.shape[0], 1, x.shape[-1])  # no batch first
        norm_x = self.norm(x)
        out = self.attention(norm_x, norm_x, norm_x,
                             need_weights=need_weights,
                             average_attn_weights=True)
        if need_weights:
            out, attention_weights = out
        x = x + out[0]
        x = x.squeeze(1)
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))

        if was_unbatched:
            x = MIL.ensure_unbatched(x)
        return x, attention_weights



class Transformer(MIL):
    def __init__(self,
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_classes: int = 2,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 num_attention_layers: int = 2,
                 num_heads: int = 8,
                 encoder_mlp_dim: int = -1,
                 ):
        super(Transformer, self).__init__(in_dim=in_dim,
                                          embed_dim=embed_dim,
                                          num_classes=num_classes)
        self.num_attention_layers = num_attention_layers
        self.dropout = dropout
        self.patch_embed = create_mlp(in_dim=in_dim,
                              hid_dims=[embed_dim] * (num_fc_layers - 1),
                              dropout=dropout,
                              out_dim=embed_dim,
                              end_with_fc=False)


        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        make_trans_layer = lambda: TransLayer(dim=embed_dim, heads=num_heads, dropout=dropout,
                                              mlp_dim=encoder_mlp_dim)
        self.blocks = nn.ModuleList(
            [make_trans_layer() for _ in range(num_attention_layers)])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.steps = 0
        self.slide_dim = embed_dim

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor)  -> torch.Tensor:
        """
        Abstract method to compute attention scores.

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.

        Returns:
            A: [B x num. attention heads x M]-dim torch.Tensor (attention scores)
               or (h_transformed, A) if attention_only is False.
        """
        pass

    def forward_features(self, h: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, dict]:
        """
        Abstract method to aggregate features using attention.

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.

        Returns:
            h: [B x D]-dim torch.Tensor, the aggregated bag-level feature.
        """
        self.steps += 1
        intermed_dict = {}
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        # h: [B, n, in_dim]
        h = self.patch_embed(h)  # [B, n, embed_dim]
        h = MIL.ensure_batched(h)
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        for i, block in enumerate(self.blocks):
            if i == 0 and return_attention:
                h, intermed_dict['attention'] = block(h, need_weights=True)
            else:
                h, _ = block(h, need_weights=False)

        # ---->cls_token
        h = MIL.ensure_unbatched(h)
        h = self.norm(h)[:1, :]

        return h, intermed_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the classification head.

        Args:
            h: [B x D]-dim torch.Tensor, the aggregated bag-level feature.

        Returns:
            logits: [B x num_classes]-dim torch.Tensor, the classification logits.
        """
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor, loss_fn: nn.Module=None,
                label: torch.LongTensor=None, attn_mask: torch.Tensor=None, return_attention: bool=False,
                return_slide_feats: bool = False) -> tuple[dict, dict]:
        """
        Complete forward pass of the Transformer MIL model.

        Args:
            h (torch.Tensor): [B x M x D]-dim tensor representing patch embeddings.
            loss_fn (nn.Module, optional): Loss function to compute classification loss.
            label (torch.LongTensor, optional): Ground truth labels.
            attn_mask (torch.Tensor, optional): Optional attention mask.
            return_attention (bool, optional): Whether to return attention weights in the log dict.

        Returns:
            results_dict (dict): Dictionary containing 'logits' and 'cls_loss'.
            log_dict (dict): Dictionary containing intermediate results, including attention and loss.
        """
        wsi_feats, log_dict = self.forward_features(h, return_attention=return_attention)  # log dict contains attention
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict

#@dataclass
class TransformerConfig(PretrainedConfig):
    model_type = 'transformer'

    def __init__(self,
        #_target_: str = "src.models.transformer.TransformerModel",
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        in_dim: int = 1024,
        num_classes: int = 2,
        num_attention_layers: int = 2,
        num_heads: int = 8,
        encoder_mlp_dim: int = -1,  # -1 means no mlp
        **kwargs):
        #self._target = _target_
        self.embed_dim = embed_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.encoder_mlp_dim = encoder_mlp_dim
        super().__init__(**kwargs)



class TransformerModel(PreTrainedModel):
    config_class = TransformerConfig

    def __init__(self, config: TransformerConfig, **kwargs):
        """
        load a model with the given config. Overwrite config attributes with any model kwargs
        """
        self.config = config
        for k,v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)
        self.model = Transformer(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_attention_layers=config.num_attention_layers,
            num_heads=config.num_heads,
            encoder_mlp_dim=config.encoder_mlp_dim,
        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier

AutoConfig.register("transformer", TransformerConfig)
AutoModel.register(TransformerConfig, TransformerModel)