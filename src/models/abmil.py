from src.models.mil_template import MIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import GlobalAttention, GlobalGatedAttention, create_mlp
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel

MODEL_TYPE = 'abmil'


class ABMIL(MIL):
    """
    ABMIL (Attention-based Multiple Instance Learning) model.

    This class implements the core ABMIL architecture, which uses a patch embedding MLP,
    followed by a global attention or gated attention mechanism, and an optional classification head.

    Args:
        in_dim (int): Input feature dimension for each instance (default: 1024).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        gate (int): Whether to use gated attention (True) or standard attention (False) (default: True).
        num_classes (int): Number of output classes for the classification head (default: 2).
    """

    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            num_classes: int = 2,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        """
        Compute the attention scores (and optionally the embedded features) for the input instances.

        Args:
            h (torch.Tensor): Input tensor of shape [B, M, D], where B is the batch size,
                M is the number of instances (patches), and D is the input feature dimension.
            attn_mask (torch.Tensor, optional): Optional attention mask of shape [B, M], where 1 indicates
                valid positions and 0 indicates masked positions. If provided, masked positions are set to
                a very large negative value before softmax.
            attn_only (bool, optional): If True, return only the attention scores (A).
                If False, return a tuple (h, A) where h is the embedded features and A is the attention scores.

        Returns:
            torch.Tensor: If attn_only is True, returns the attention scores tensor of shape [B, K, M],
                where K is the number of attention heads (usually 1). If attn_only is False, returns a tuple
                (h, A) where h is the embedded features of shape [B, M, D'] and A is the attention scores.
        """
        h = self.patch_embed(h)
        A = self.global_attn(h)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base if return_attention else None}
        return h, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x D]-dim torch.Tensor.

        Returns:
            logits: [B x num_classes]-dim torch.Tensor.
        """
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> torch.Tensor:
        """
        Forward pass for ABMIL.

        Args:
            h: [B, M, D] input features.
            loss_fn: Optional loss function.
            label: Optional labels.
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (results_dict, log_dict) with logits and loss.
        """
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict


class ABMILGatedBaseConfig(PretrainedConfig):
    """
    Configuration class for the ABMIL Gated Base model.

    This class stores the configuration parameters required to instantiate an ABMIL model
    with gated attention. It is compatible with Hugging Face's Transformers library and
    can be used to save, load, and share model configurations.

    Args:
        gate (bool): Whether to use gated attention (default: True).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        in_dim (int): Input feature dimension for each instance (default: 1024).
        num_classes (int): Number of output classes for the classification head (default: 2).
        **kwargs: Additional keyword arguments passed to the PretrainedConfig base class.

    Attributes:
        model_type (str): The model type identifier ("abmil").
        gate (bool): Whether to use gated attention.
        embed_dim (int): Embedding dimension after patch embedding.
        attn_dim (int): Dimension of the attention mechanism.
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP.
        dropout (float): Dropout rate applied in the MLP and attention layers.
        in_dim (int): Input feature dimension for each instance.
        num_classes (int): Number of output classes for the classification head.
        auto_map (dict): Mapping for Hugging Face AutoConfig and AutoModel registration.
    """

    model_type = MODEL_TYPE

    # add mapping

    # _target_: str = "src.models.abmil.ABMIL"
    def __init__(self,
                 gate: bool = True,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 in_dim: int = 1024,
                 num_classes: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.gate = gate
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.auto_map = {
            "AutoConfig": "modeling_abmil.ABMILGatedBaseConfig",
            "AutoModel": "modeling_abmil.ABMILModel",
        }


class ABMILModel(PreTrainedModel):
    config_class = ABMILGatedBaseConfig

    def __init__(self, config: ABMILGatedBaseConfig, **kwargs):
        """
        Initialize ABMILModel with the given config, allowing attribute overrides via kwargs.
        """

        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)
        self.model = ABMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate,
            num_classes=config.num_classes
        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(ABMILGatedBaseConfig.model_type, ABMILGatedBaseConfig)
AutoModel.register(ABMILGatedBaseConfig, ABMILModel)

