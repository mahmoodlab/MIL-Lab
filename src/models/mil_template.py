import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from abc import ABC, abstractmethod


class MIL(ABC, nn.Module):
    """
    Abstract base class for MIL (Multiple Instance Learning) models.
    Defines the core forward pass methods that MIL implementations must provide.
    """

    def __init__(self,
                 in_dim: int,
                 embed_dim: int,
                 num_classes: int,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward_attention(self, h: torch.Tensor)  -> torch.Tensor:
        """
        Method to compute attention scores.

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.
            : Optional attention mask.
            attn_only: If True, returns only attention scores; otherwise, returns embeddings and scores.

        Returns:
            A: [B x num. attention heads x M]-dim torch.Tensor (attention scores)
               or (h_transformed, A) if attn_only is False.
        """
        pass

    @abstractmethod
    def forward_features(self, h: torch.Tensor, return_attention: bool=False) -> tuple[torch.Tensor, dict]:
        """
        Aggregate patch features using attention into slide-level features.

        Args:
            h: [B x M x D]-dim torch.Tensor representing patch embeddings.
            return_attention: bool indicating whether to return attention scores in intermed dict.

        Returns:
            h: [B x D]-dim torch.Tensor, the aggregated bag-level feature.
            intermeds: dict containing intermediate results (optional, can be extended by concrete implementations).
        """
        pass

    @abstractmethod
    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply classification head to the aggregated slide-level feature.

        Args:
            h: [B x D]-dim torch.Tensor, the aggregated bag-level feature.

        Returns:
            logits: [B x num_classes]-dim torch.Tensor, the classification logits.
        """
        pass

    def forward(self, h: Union[torch.Tensor, list, tuple],
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> tuple[dict, dict]:
        """
        Complete forward pass of the model. 
        Supports hierarchical input (list of slides) for Late Fusion.

        Args:
            h: [B x M x D]-dim torch.Tensor or List of [M_i x D] tensors.
            loss_fn: Optional loss function.
            label: Optional ground truth labels.
            attn_mask: Optional attention mask (only for non-hierarchical).
            return_attention: If True, return attention scores in log_dict.
            return_slide_feats: If True, return 'slide_feats' in log_dict.

        Returns:
            results_dict: A dictionary containing the 'logits' and 'loss'
            log_dict: An dictionary optionally containing attention and intermediate results
        """
        if isinstance(h, (list, tuple)):
            # Late Fusion: Process each slide independently and average embeddings
            slide_feats = []
            attentions = []
            for slide_h in h:
                # Ensure slide is batched [1, M, D]
                slide_h = self.ensure_batched(slide_h)
                s_feat, s_log = self.forward_features(slide_h, return_attention=return_attention)
                slide_feats.append(s_feat)
                if return_attention:
                    attentions.append(s_log.get('attention'))
            
            # Average slide embeddings: [num_slides, B, D] -> [B, D]
            # (Assuming B=1 for hierarchical data loaders currently)
            wsi_feats = torch.stack(slide_feats).mean(dim=0)
            log_dict = {'attention': attentions if return_attention else None}
        else:
            # Standard path
            wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
            
        logits = self.forward_head(wsi_feats)
        cls_loss = self.compute_loss(loss_fn, logits, label)
        
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
            
        return results_dict, log_dict

    @staticmethod
    def ensure_batched(tensor: torch.Tensor, return_was_unbatched: bool = False) -> torch.Tensor:
        """
        Ensure the tensor is batched (has a batch dimension).

        Args:
            tensor: A torch.Tensor that may or may not have a batch dimension.

        Returns:
            A batched torch.Tensor.
        """
        was_unbatched = False
        while len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
            was_unbatched = True
        if return_was_unbatched:
            return tensor, was_unbatched
        return tensor

    @staticmethod
    def ensure_unbatched(tensor: torch.Tensor, return_was_batched: bool = False) -> torch.Tensor:
        """
        Ensure the tensor is unbatched (removes the batch dimension if present).

        Args:
            tensor: A torch.Tensor that may or may not have a batch dimension.

        Returns:
            An unbatched torch.Tensor.
        """
        was_batched = True
        while len(tensor.shape) > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            was_batched = False
        if return_was_batched:
            return tensor, was_batched
        return tensor

    @staticmethod
    def compute_loss(loss_fn: nn.Module,
                     logits: torch.Tensor,
                     label: torch.LongTensor) -> torch.Tensor:
        """
        Compute the loss using the provided loss function.

        Args:
            loss_fn: A callable loss function.
            logits: The model's output logits.
            label: The ground truth labels.

        Returns:
            A scalar tensor representing the computed loss.
        """
        if loss_fn is None or logits is None:
            return None
        return loss_fn(logits, label)

    def initialize_weights(self):
        """
        Initialize the weights of the model with kaiming he for linear layers, and xavier for all others
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)


    def initialize_classifier(self, num_classes: Optional[int] = None):
        """
        Initialize the classifier layer
        """
        if num_classes is None:
            num_classes = self.num_classes
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity='relu')
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)





