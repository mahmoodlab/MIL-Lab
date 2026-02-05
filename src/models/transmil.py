import torch
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention
from src.models.layers import create_mlp
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
from src.models.mil_template import MIL

class TransLayer(nn.Module):
    def __init__(self, norm_layer: nn.Module = nn.LayerNorm, dim: int = 512, num_heads: int = 8):
        """
        Transformer Layer with Nystrom Attention.

        Args:
            norm_layer (nn.Module): Normalization layer, default is nn.LayerNorm.
            dim (int): Dimension for the transformer layer, default is 512.
        """
        super().__init__()
        self.norm = norm_layer(dim)
        self.attention = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and normalization.
        """
        x = x + self.attention(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim: int = 512):
        """
        Position-wise Projection Embedded Gradient (PPEG) for positional encoding.

        Args:
            dim (int): Dimension for the embedding, default is 512.
        """
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass for the PPEG layer.

        Args:
            x (torch.Tensor): Input tensor.
            H (int): Height for reshaping.
            W (int): Width for reshaping.

        Returns:
            torch.Tensor: Output tensor with positional encoding applied.
        """
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(MIL):
    def __init__(self, in_dim: int, embed_dim: int,
                 num_fc_layers: int, dropout: float,
                 num_attention_layers: int, num_classes: int, num_heads: int=8):
        """
        TransMIL model with transformer-based Multi-instance Learning.

        Args:
            in_dim (int): Input dimension for the MLP.
            embed_dim (int): Embedding dimension for all layers.
            n_fc_layers (int): Number of fully connected layers in the MLP.
            dropout (float): Dropout rate for MLP.
            n_attention_layers (int): Number of transformer attention layers.
            n_classes (int): Number of output classes for classification.
        """
        super(TransMIL, self).__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed: nn.Module = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        self.pos_layer: nn.Module = PPEG(dim=embed_dim)
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.blocks: nn.ModuleList = nn.ModuleList(
            [TransLayer(dim=embed_dim, num_heads=num_heads) for _ in range(num_attention_layers)]
        )

        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.classifier: nn.Linear = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor) -> torch.Tensor:
        pass

    def forward_features(self, h: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Get slide-level features from cls token.

        Args:
            h (torch.Tensor): The input tensor of shape (features, dim) or
                              (batch_size, features, dim).

        Returns:
            torch.Tensor: Slide-level feature of cls token. Output shape will be
                          (1, 1, embed_dim) if input was 2D or
                          (batch_size, 1, embed_dim) if input was 3D.
        """
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        h = self.patch_embed(h)
        h, h_square, w_square = self._square_pad(h)
        h = self._add_cls_token(h)
        h, attn = self._apply_trans_layers(h, h_square, w_square, return_attention)
        wsi_feat = self.norm(h)[:, 0]  # get cls token
        return wsi_feat, attn

    def _apply_trans_layers(self, h: torch.Tensor, h_square: int, w_square: int, return_attention: bool = False) -> torch.Tensor:
        """
        Apply transformer layers to the input.

        Args:
            h (torch.Tensor): Input tensor after adding class token.
            h_square (int): Square height obtained from padding calculation.
            w_square (int): Square width obtained from padding calculation.
            return_attention (bool): whether to compute attention scores wrt cls token

        Returns:
            torch.Tensor: Transformed tensor.
        """
        intermed_dict = {}
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 0:
                if return_attention:
                    # compute attention scores wrt cls token in first position
                    cls_token = h[:, 0]  # b x d
                    feats = h[:, 1:] # b x n x d
                    # compute the dot prod similarity between each feat and cls token
                    intermed_dict['attention'] = torch.matmul(feats, cls_token.unsqueeze(-1)).squeeze(-1)
                h = self.pos_layer(h, h_square, w_square)
        return h, intermed_dict

    def _square_pad(self, h: torch.Tensor) -> tuple:
        """
        Pad feature tensor to make it square.

        Args:
            h (torch.Tensor): Input tensor.

        Returns:
            tuple: Padded tensor, square height, and square width.
        """
        H = h.shape[1]
        add_length, h_square, w_square = self._get_square_length(H)
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        return h, h_square, w_square

    def _add_cls_token(self, h: torch.Tensor) -> torch.Tensor:
        """
        Add class token to the input tensor.

        Args:
            h (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Input tensor with class token added.
        """
        B, H = h.shape[0], h.shape[1]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        return h

    def _get_square_length(self, H: int) -> tuple:
        """
        Calculate the required lengths to convert the input into square form.

        Args:
            H (int): Original height (or length) of the input tensor.

        Returns:
            tuple: Additional length needed, and new square dimensions.
        """
        h_square, w_square = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = h_square * w_square - H
        return add_length, h_square, w_square

    def forward_head(self, wsi_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.

        Args:
            wsi_feat (torch.Tensor): Slide-level feature for classification.

        Returns:
            torch.Tensor: Logits for classification.
        """
        logits = self.classifier(wsi_feat)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module=None,
                label: torch.LongTensor=None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> tuple:
        """
        Complete forward pass of the model.

        Args:
            h (torch.Tensor): Input feature tensor.

        Returns:
            tuple: Slide-level features and logits from the classifier.
        """
        wsi_feats, intermeds = self.forward_features(h, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = self.compute_loss(loss_fn=loss_fn, label=label, logits=logits)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict = {'loss': cls_loss.item() if cls_loss is not None else -1}
        if return_attention:
            log_dict['attention'] = intermeds['attention']
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict



#@dataclass
class TransMILConfig(PretrainedConfig):
    model_type = 'transmil'

    def __init__(self,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        dropout: float = 0.25,
        in_dim: int = 1024,
        num_classes: int = 2,
        num_attention_layers: int = 2,
        num_heads: int = 4,
        **kwargs
        ):
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_fc_layers = num_fc_layers
        self.dropout = dropout
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        super().__init__(**kwargs)


class TransMILModel(PreTrainedModel):
    config_class = TransMILConfig

    def __init__(self, config: TransMILConfig, **kwargs):
        """
        load a model with the given config. Overwrite config attributes with any model kwargs
        """
        self.config = config
        super().__init__(config)
        for k,v in kwargs.items():
            setattr(config, k, v)

        self.model = TransMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_heads=config.num_heads,
            num_attention_layers=config.num_attention_layers
        )
        self.forward = self.model.forward
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier



AutoConfig.register("transmil", TransMILConfig)
AutoModel.register(TransMILConfig, TransMILModel)