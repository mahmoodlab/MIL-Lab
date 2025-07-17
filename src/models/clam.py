import torch
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from dataclasses import dataclass

from src.models.mil_template import MIL
from src.models.layers import GlobalGatedAttention, GlobalAttention, create_mlp
from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig

# Define the core CLAMSB model
class CLAMSB(MIL):
    """
    Clustering-constrained Attention Multiple Instance Learning (CLAM) - Single-Branch version.
    This model integrates instance-level and bag-level learning through an attention mechanism
    and instance classifiers.
    """
    
    

    def __init__(self,
                 in_dim: int = 1024,  # Input dimension of each instance feature
                 embed_dim: int = 512,  # Dimension after instance embedding MLP
                 n_fc_layers: int = 1,  # Number of layers in the instance embedding MLP
                 dropout: float = 0.25,  # Dropout rate for MLP and attention
                 gate: bool = True,  # Whether to use gated attention
                 attention_dim: int = 384,  # Dimension for the attention network
                 num_classes: int = 2,  # Number of output classes for bag-level prediction
                 k_sample: int = 8,  # Number of top/bottom patches to sample for instance-level training
                 subtyping: bool = False,  # Whether this is a subtyping problem (affects inst_eval_out)
                 instance_loss_fn: str = 'svm',  # Loss function type for instance-level prediction ('ce' or 'svm')
                 bag_weight: float = 0.7):  # Weight for bag-level loss vs. instance-level loss

        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)

        self.k_sample = k_sample
        self.subtyping = subtyping
        self.bag_weight = bag_weight

        # Instance Embedding MLP: Transforms input features to 'embed_dim'
        self.patch_embed = create_mlp(in_dim=in_dim,
                              hid_dims=[embed_dim] * (n_fc_layers - 1),
                              dropout=dropout,
                              out_dim=embed_dim,
                              end_with_fc=False)

        # Attention Network: Calculates attention scores for instances
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(L=embed_dim,
                                       D=attention_dim,
                                       dropout=dropout,
                                       num_classes=1)  # Single attention head

        # Bag-level Classifier: Maps aggregated features to class logits
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Instance Classifiers: One classifier per class for instance-level prediction
        instance_classifiers = [nn.Linear(embed_dim, 2) for _ in range(num_classes)]  # Binary classifier per class
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        # Set instance loss function
        if instance_loss_fn == 'svm':
            # Note: SmoothTop1SVM would need to be imported and potentially moved to a device if used.
            # For this example, assuming it's available or handling device.
            try:
                from topk.svm import SmoothTop1SVM 
            except:
                print(f'SmoothTop1SVM not found. '
                      f'Please install it via pip: pip install git+https://github.com/oval-group/smooth-topk')
            # pip install git+https://github.com/oval-group/smooth-topk
            try:
                self.instance_loss_fn = SmoothTop1SVM(n_classes=2).cuda()
            except AssertionError:
                raise AssertionError('SmoothTop1SVM requires CUDA to be available. If not available, please use the ce loss function')
        else:
            self.instance_loss_fn = nn.CrossEntropyLoss()
        self.initialize_weights()

    @staticmethod
    def create_positive_targets(length: int, device: torch.device) -> torch.LongTensor:
        """Helper to create a tensor of positive (1) labels."""
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length: int, device: torch.device) -> torch.LongTensor:
        """Helper to create a tensor of negative (0) labels."""
        return torch.full((length,), 0, device=device).long()

    def _check_inputs(self, features, loss_fn, label):
        if features.dim() == 3 and features.shape[0] > 1:
            raise ValueError(f'CLAM does not currently support batch size > 1')
        if label is None:
            raise ValueError(f'Label must be provided for CLAM')
        if loss_fn is None:
            raise ValueError("Loss function must be provided")

    def inst_eval(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Instance-level evaluation for in-the-class attention branch (positive instances).

        Args:
            A (torch.Tensor): Attention scores (N,).
            h (torch.Tensor): Instance features (N, D).
            classifier (nn.Module): Instance-level classifier.

        Returns:
            tuple: (instance_loss, predictions, targets)
        """
        # Ensure A is 2D for torch.topk consistency
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # Select top-k positive instances based on attention scores
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        # Select top-k negative instances (lowest attention)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        # Create targets for selected instances
        p_targets = self.create_positive_targets(self.k_sample, h.device)
        n_targets = self.create_negative_targets(self.k_sample, h.device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)

        # Concatenate selected instances and get logits
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)

        # Get predictions and compute loss
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A: torch.Tensor, h: torch.Tensor, classifier: nn.Module) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Instance-level evaluation for out-of-the-class attention branch (negative instances).
        Used primarily in subtyping problems.

        Args:
            A (torch.Tensor): Attention scores (N,).
            h (torch.Tensor): Instance features (N, D).
            classifier (nn.Module): Instance-level classifier.

        Returns:
            tuple: (instance_loss, predictions, targets)
        """
        # Ensure A is 2D
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # Select top-k instances, which are treated as negative for out-of-class evaluation
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)

        # Targets are negative for these instances
        p_targets = self.create_negative_targets(self.k_sample, h.device)

        # Get logits and compute loss
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward_attention(self, h: torch.Tensor, attention_only: bool = False) -> Union[torch.Tensor, tuple[
        torch.Tensor, torch.Tensor]]:
        """
        Processes instance features through the instance embedding and attention network.

        Args:
            h (torch.Tensor): Input instance features of shape [B, M, D].
                              (Note: CLAM typically processes one bag at a time, so B=1 initially).
            attention_only (bool): If True, returns only attention scores.

        Returns:
            torch.Tensor: Attention scores [1, M] if attention_only=True.
                          Or (h_embedded [M, E], A [1, M]) if attention_only=False.
        """
        # Apply instance embedding
        # h should be [M, D] after squeezing batch dimension if input is [1, M, D]
        h = self.patch_embed(h.squeeze(0))  # Squeeze batch dimension for CLAM's attention logic if input is [1, M, D]

        # Compute attention scores
        A = self.global_attn(h)  # Output A: [M x 1] (or M x K if multiple heads)
        A = torch.transpose(A, 1, 0)  # Transpose to [1 x M] for consistency with CLAM's original logic

        if attention_only:
            return A
        else:
            return h, A  # h (embedded instances) [M, E], A (unnormalized attention) [1, M]

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Applies the bag-level classification head.

        Args:
            h (torch.Tensor): Aggregated bag-level features [B, E].

        Returns:
            logits (torch.Tensor): Raw classification logits [B, num_classes].
        """
        logits = self.classifier(h)
        return logits

    
    def forward_features(self, h: torch.Tensor, return_attention: bool = True) -> tuple[
        torch.Tensor, dict]:
        """
        Aggregates instance features into a single bag-level feature vector using attention.

        Args:
            h (torch.Tensor): Input instance features of shape [B, M, D].
            return_attention (bool): returns attention scores always
        Returns:
            tuple:
                - M (torch.Tensor): Aggregated bag-level features [B, E].
                - log_dict (dict): Dict storing 'attention' and 'instance_feats'.
                instance_feats (torch.Tensor): Embedded instance features [M, E] (if return_attention=True). attention (torch.Tensor): Unnormalized attention scores [1, M] (if return_attention=True).
        """
        # Get embedded instance features and unnormalized attention scores
        h_embedded, attention = self.forward_attention(h)  # h_embedded: [M, E], attention: [1, M]
        log_dict = {'instance_feats': h_embedded}
        if return_attention:
            log_dict['attention'] = attention

        # Apply softmax to attention scores for weighting
        attention_scaled = F.softmax(attention, dim=-1)  # Softmax over instances (M)

        M = torch.mm(attention_scaled, h_embedded)  # Aggregated bag feature [1, E]


        return M, log_dict  # M [B, E], h_embedded [M, E], attention [1, M]


    def forward_instance_heads(self, h: torch.Tensor, attention_scores: torch.Tensor,
                               label: torch.LongTensor = None) -> Union[torch.Tensor, None]:
        """
        Computes instance-level loss.

        Args:
            h (torch.Tensor): Embedded instance features [M, E].
            attention_scores (torch.Tensor): Unnormalized attention scores [1, M].
            label (torch.LongTensor, optional): True label for the bag [B].

        Returns:
            torch.Tensor | None: Total instance loss, or None if label is not provided.
        """
        if label is None:
            return None
        total_inst_loss = 0.0
        # Convert scalar label to one-hot for instance classifier selection
        # Ensure the one-hot tensor is on the same device as the input label
        inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze(0).to(label.device)

        for i in range(len(self.instance_classifiers)):
            inst_label_for_class = inst_labels[i].item()  # Get 0 or 1 for current class 'i'
            classifier = self.instance_classifiers[i]

            if inst_label_for_class == 1:  # If the bag belongs to class 'i'
                instance_loss, _, _ = self.inst_eval(attention_scores, h, classifier)
                total_inst_loss += instance_loss
            else:  # If the bag does not belong to class 'i'
                if self.subtyping:  # Only evaluate out-of-class instances for subtyping problems
                    instance_loss, _, _ = self.inst_eval_out(attention_scores, h, classifier)
                    total_inst_loss += instance_loss
                # Else: do nothing if not subtyping and bag is not of this class

        # Average loss only if subtyping or if total_inst_loss was incremented
        if self.subtyping and len(self.instance_classifiers) > 0:
            total_inst_loss /= len(self.instance_classifiers)
        elif not self.subtyping and inst_labels.sum().item() > 0:  # If there was at least one positive class
            total_inst_loss /= inst_labels.sum().item()  # Average over positive classes only
        elif total_inst_loss == 0 and inst_labels.sum().item() == 0:  # No positive classes, and not subtyping
            return None  # No instance loss to compute

        return total_inst_loss

    def compute_total_loss(self, logits: torch.Tensor, label: torch.LongTensor, loss_fn: nn.Module,
                           inst_loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined bag-level and instance-level loss.

        Args:
            logits (torch.Tensor): Bag-level raw output scores.
            label (torch.LongTensor): True bag-level label.
            loss_fn (nn.Module): Bag-level loss function.
            inst_loss (torch.Tensor): Computed instance-level loss.

        Returns:
            torch.Tensor: The total combined loss.
        """
        cls_loss = self.compute_loss(loss_fn, logits, label)
        if inst_loss is not None:
            # Combine bag-level and instance-level losses with a predefined weight
            loss = cls_loss * self.bag_weight + (1 - self.bag_weight) * inst_loss
        else:
            loss = cls_loss
        return loss

    def forward(self, h: torch.Tensor, label: torch.LongTensor = None,
                loss_fn: nn.Module = None, attn_mask = None, return_attention: bool = True,
                return_slide_feats: bool = None
                ) -> tuple[dict, dict]:
        """
        Full forward pass through the CLAMSB model.

        Args:
            h (torch.Tensor): Input instance features of shape [B, M, D].
            label (torch.LongTensor, optional): True label for the bag [B].
            loss_fn (nn.Module, optional): The bag-level loss function.

        Returns:
            tuple[dict, dict]: A tuple containing:
                - results_dict (dict): Dictionary with 'logits' and 'loss'
                - log_dict (dict): Dictionary with 'instance_loss', 'cls_loss', 'loss'.
        """
        self._check_inputs(h, loss_fn, label)
        # Get bag-level features, embedded instance features, and unnormalized attention scores
        # slide_feats: [B, E], instance_feats: [M, E], attention_scores: [1, M] (assuming B=1)
        slide_feats, intermeds = self.forward_features(h, return_attention=return_attention)

        # Get bag-level classification logits
        logits = self.forward_head(slide_feats)

        # Compute instance-level loss
        inst_loss = self.forward_instance_heads(intermeds['instance_feats'], intermeds['attention'], label)

        # Compute total combined loss
        total_loss = self.compute_total_loss(logits, label, loss_fn, inst_loss)

        # Prepare log dictionary
        log_dict = {
            'instance_loss': inst_loss.item() if inst_loss is not None else -1,
            'cls_loss': loss_fn(logits, label).item() if loss_fn is not None and label is not None else -1,
            'loss': total_loss.item()
        }

        results_dict = {'logits': logits, 'loss': total_loss}
        if return_attention:
            log_dict['attention'] = intermeds['attention']
        if return_slide_feats:
            log_dict['slide_feats'] = slide_feats


        return results_dict, log_dict


# Define the Hugging Face compatible configuration class for CLAM
@dataclass
class CLAMConfig(PretrainedConfig):
    """
    Configuration class for CLAMModel.

    Args:
        _target_ (str): Path to the model class for instantiation.
        gate (bool): Whether to use gated attention.
        in_dim (int): Input dimension of each instance feature.
        num_classes (int): Number of output classes for bag-level prediction.
        embed_dim (int): Dimension after instance embedding MLP.
        attention_dim (int): Dimension for the attention network.
        n_fc_layers (int): Number of layers in the instance embedding MLP.
        dropout (float): Dropout rate for MLP and attention.
        k_sample (int): Number of top/bottom patches to sample for instance-level training.
        subtyping (bool): Whether this is a subtyping problem.
        instance_loss_fn (str): Loss function type for instance-level prediction ('ce' or 'svm').
        bag_weight (float): Weight for bag-level loss vs. instance-level loss.
    """

    model_type = 'clam'

    def __init__(self, 
        gate: bool = True,
        in_dim: int = 1024,
        num_classes: int = 2,
        embed_dim: int = 512,
        attention_dim: int = 384,
        n_fc_layers: int = 1,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = False,
        instance_loss_fn: str = 'svm',
        bag_weight: float = 0.7,
        **kwargs):
            super().__init__(**kwargs)
            self.gate = gate
            self.in_dim = in_dim
            self.num_classes = num_classes
            self.embed_dim = embed_dim
            self.attention_dim = attention_dim
            self.n_fc_layers = n_fc_layers
            self.dropout = dropout
            self.k_sample = k_sample
            self.subtyping = subtyping
            self.instance_loss_fn = instance_loss_fn
            self.bag_weight = bag_weight

# Define the Hugging Face compatible PreTrainedModel wrapper
class CLAMModel(PreTrainedModel):
    """
    Hugging Face PreTrainedModel wrapper for the CLAMSB model.
    This class handles loading configurations and instantiating the core CLAMSB module.
    """
    config_class = CLAMConfig  # Link to the custom configuration class

    def __init__(self, config: CLAMConfig, **kwargs):
        """
        Initializes the CLAMModel.

        Args:
            config (CLAMConfig): The configuration object for the model.
            **kwargs: Additional keyword arguments to override config attributes.
        """
        # Update config attributes with any provided kwargs
        self.config = config
        for k, v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)  # Initialize the PreTrainedModel base class

        # Instantiate the core CLAMSB model using parameters from the config
        self.model = CLAMSB(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            n_fc_layers=config.n_fc_layers,
            dropout=config.dropout,
            gate=config.gate,
            attention_dim=config.attention_dim,
            num_classes=config.num_classes,
            k_sample=config.k_sample,
            subtyping=config.subtyping,
            instance_loss_fn=config.instance_loss_fn,
            bag_weight=config.bag_weight
        )

        # Directly expose the core model's methods for convenience
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register(CLAMConfig.model_type, CLAMConfig)
AutoModel.register(CLAMConfig, CLAMModel)


