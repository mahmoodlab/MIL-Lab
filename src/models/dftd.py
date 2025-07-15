import numpy as np
import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
import torch.nn.functional as F
import os
import torch.nn as nn
import torch
import random
#from utils import get_cam_1d
from src.models.layers import create_mlp, GlobalGatedAttention
from src.models.abmil import ABMIL
from src.models.mil_template import MIL
from dataclasses import dataclass

from src.builder_utils import _cfg, build_model_with_cfg
_model_default_cfgs = {
    'default': _cfg(),
}


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.global_attn = GlobalGatedAttention(L, D, num_classes=K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x): ## x: N x L
        """
        Forward pass for the attention-with-classifier module.

        Args:
            x (torch.Tensor): Input feature tensor of shape (N, L).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - pred (torch.Tensor): The predicted logits for the WSI.
                - afeat (torch.Tensor): The attention-weighted features.
                - AA (torch.Tensor): The attention weights.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        AA = self.global_attn(x)  # (N, 1) or (N,) attention weights
        if AA.dim() == 1:
            AA = AA.unsqueeze(-1)  # (N, 1)
        afeat = torch.sum(AA * x, dim=0, keepdim=True)  # weighted sum: (1, D)
        pred = self.classifier(afeat)  # (1, num_cls)
        return pred, afeat, AA

class DFTD(MIL):
    def __init__(self,
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 attention_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.25,
                 attention_drop: float = 0.25,
                 distill: str = 'MaxMinS',
                 num_group: int = 8,
                 total_instance: int = 8,
                 bag_weight: float = 0.7,
                 num_residuals: int = 0,
                 ):
        super(DFTD, self).__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.classifier = Classifier_1fc(embed_dim, num_classes, droprate=dropout)  #.to(device)
        self.attention = GlobalGatedAttention(L=embed_dim, D=attention_dim, dropout=attention_drop)
        self.attCls = Attention_with_Classifier(L=embed_dim, D=attention_dim, K=1, num_cls=num_classes)
        self.distill = distill
        self.num_group = num_group
        self.total_instance = total_instance
        self.inst_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bag_weight = bag_weight
        self.patch_embed = DimReduction(in_dim, embed_dim, num_residuals)
        self.initialize_weights()

    def forward(self, h: torch.Tensor, label: torch.LongTensor = None,
                loss_fn: nn.Module = None, return_attention: bool = False,
                return_slide_feats: bool = False) -> tuple[dict, dict]:
        """
        Forward pass for the DFTD model.

        Args:
            h (torch.Tensor): Input feature tensor of shape (N, in_dim).
            label (torch.LongTensor, optional): Ground truth label tensor for the slide. Required.
            loss_fn (nn.Module, optional): Loss function to compute classification loss. Required.
            return_attention (bool, optional): Whether to return attention weights in the log dict.

        Returns:
            results_dict (dict): Dictionary containing 'logits', 'loss', and 'instance_loss'.
            log_dict (dict): Dictionary containing 'instance_loss', 'cls_loss', 'loss', and optionally 'attention'.
        """
        assert label is not None, "label is required for DFTD"
        assert loss_fn is not None, "loss_fn is required for DFTD"

        #h = h.squeeze(0)  # no batch dim
        h = self.ensure_unbatched(h)
        slide_pseudo_feat, intermed_dict = self.forward_features(h, label)
        logits, attention = self.forward_head(slide_pseudo_feat)
        slide_sub_preds = intermed_dict['slide_sub_preds']
        slide_sub_labels = intermed_dict['slide_sub_labels']

        instance_loss = self.get_instance_loss(slide_sub_preds, slide_sub_labels)
        cls_loss = self.compute_loss(loss_fn, logits, label)
        loss = self.get_total_loss(cls_loss, instance_loss)

        log_dict = {'instance_loss': instance_loss.item(), 'cls_loss': cls_loss.item(), 'loss': loss.item()}
        results_dict = {'logits': logits, 'loss': loss.to(h.dtype).to(h.device), 'instance_loss': instance_loss}
        if return_attention:
            log_dict['attention'] = attention
        if return_slide_feats:
            log_dict['slide_feats'] = slide_pseudo_feat
        return results_dict, log_dict

    def get_total_loss(self, cls_loss: torch.Tensor, inst_loss: torch.Tensor):
        """
        Compute the total loss as a weighted sum of the classification loss and the instance loss.

        Args:
            cls_loss (torch.Tensor): The classification loss tensor.
            inst_loss (torch.Tensor): The instance loss tensor.

        Returns:
            loss (torch.Tensor): The total loss, combining classification and instance loss.
        """
        if inst_loss is not None:
            loss = cls_loss * self.bag_weight + (1 - self.bag_weight) * inst_loss
            #loss = cls_loss * 0.7 + (1 - 0.7) * inst_loss
        else:
            loss = cls_loss
        return loss

    def get_instance_loss(self, slide_sub_preds: torch.Tensor, slide_sub_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the instance loss as the mean of the cross-entropy loss between the predicted and ground truth labels.

        Args:
            slide_sub_preds (torch.Tensor): The predicted logits for the slide.
            slide_sub_labels (torch.Tensor): The ground truth labels for the slide.

        Returns:
            torch.Tensor: The instance loss
        """
        instance_loss = self.inst_loss(slide_sub_preds, slide_sub_labels.squeeze(-1)).mean()
        return instance_loss

    def forward_head(self, wsi_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward the WSI features through the attention-based classifier head.

        Args:
            wsi_feats (torch.Tensor): The input features for the WSI (whole slide image).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - logits (torch.Tensor): The predicted logits for the WSI.
                - attention (torch.Tensor): The attention weights.
        """
        logits, slide_feats, attention = self.attCls(wsi_feats)
        return logits, attention

    def forward_attention(self, h: torch.Tensor) -> torch.Tensor:
        pass


    def get_cam_1d(
        self,
        classifier: torch.nn.Module,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 1D class activation maps (CAM) for the given features using the classifier's weights.

        Args:
            classifier (torch.nn.Module): The classifier module whose penultimate layer weights are used for CAM computation.
            features (torch.Tensor): The input features of shape (batch, group, feature_dim).

        Returns:
            torch.Tensor: The computed CAM maps of shape (batch, class, group).
        """
        tweight = list(classifier.parameters())[-2]
        cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
        return cam_maps


    def forward_features(
        self,
        h: torch.Tensor,
        label: torch.Tensor,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute pseudobag slide-level features and instance predictions.

        This method splits the input feature tensor into groups, applies dimensionality reduction,
        attention, and classification to each group, and aggregates the results according to the
        distillation strategy.

        Args:
            h (torch.Tensor): Input feature tensor of shape (num_instances, feature_dim).
            label (torch.Tensor): Slide-level label tensor.
            return_attention (bool, optional): Unused, for interface compatibility.

        Returns:
            Tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - slide_pseudo_feat (torch.Tensor): Aggregated pseudo-bag features for the slide.
                - intermed_dict (dict): Dictionary containing:
                    - 'slide_sub_preds' (torch.Tensor): Subgroup predictions.
                    - 'slide_sub_labels' (torch.Tensor): Subgroup labels.
        """
        instance_per_group = self.total_instance // self.num_group

        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        # Process feature tensor
        feat_index = list(range(h.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.num_group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        for tindex in index_chunk_list:
            slide_sub_labels.append(label.unsqueeze(0))
            subFeat_tensor = torch.index_select(h, dim=0, index=torch.LongTensor(tindex).to(h.device))

            # Dimensionality reduction
            tmidFeat = self.patch_embed(subFeat_tensor)
            if len(tmidFeat.shape) == 3:
                tmidFeat = tmidFeat.squeeze(0)

            # Attention mechanism
            tAA = self.attention(tmidFeat)
            if len(tAA.shape) == 2:
                tAA = tAA.squeeze(-1)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)

            # Classifier prediction
            tPredict = self.classifier(tattFeat_tensor)
            slide_sub_preds.append(tPredict)

            # Patch-level prediction logits
            patch_pred_logits = self.get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)

            # Select top-k indices
            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            # MaxMin instance features
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            # Choose distillation strategy
            if self.distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif self.distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif self.distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)

        # Return the sub-predictions and pseudo features for the next layer
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)
        intermed_dict = {'slide_sub_preds': slide_sub_preds, 'slide_sub_labels': slide_sub_labels}
        return slide_pseudo_feat, intermed_dict


    def reset_classifier(self):
        self.classifier.fc.reset_parameters()
        self.attCls.classifier.reset_parameters()


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.in_features = n_channels
        self.out_features = n_classes
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):
        """
        Forward pass for the 1-layer classifier.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch, feature_dim).

        Returns:
            torch.Tensor: The output logits of shape (batch, num_classes).
        """
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0, dropout=0):
        super(DimReduction, self).__init__()

        self.fc1 = create_mlp(in_dim=n_channels, hid_dims=[], dropout=dropout,
                     out_dim=m_dim, end_with_fc=False, bias=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        """
        Forward pass for the dimensionality reduction module.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch, feature_dim).

        Returns:
            torch.Tensor: The output feature tensor of shape (batch, feature_dim).
        """
        x = self.fc1(x)
        if self.numRes > 0:
            x = self.resBlocks(x)
        return x


@dataclass
class DFTDConfig(PretrainedConfig):
    model_type = 'dftd'

    def __init__(self,
        in_dim: int = 1024,
        embed_dim: int = 512,
        attention_dim: int = 128,
        num_residuals: int = 4,
        num_fc_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.25,
        attention_drop: float = 0.25,
        num_group: int = 8,
        total_instance: int = 4,
        bag_weight: float = 0.7,
        distill: str = 'MaxMinS',
        **kwargs
    ):
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_residuals = num_residuals
        self.num_fc_layers = num_fc_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.attention_drop = attention_drop
        self.num_group = num_group
        self.total_instance = total_instance
        self.bag_weight = bag_weight
        self.distill = distill
        self.attention_dim = attention_dim
        super().__init__(**kwargs)


class DFTDModel(PreTrainedModel):
    config_class = DFTDConfig

    def __init__(self, config: DFTDConfig, **kwargs):
        """
        load a model with the given config. Overwrite config attributes with any model kwargs
        """
        self.config = config
        for k,v in kwargs.items():
            setattr(config, k, v)

        super().__init__(config)  # tells hf how to initialize
        self.model = DFTD(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            attention_dim=config.attention_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
            attention_drop=config.attention_drop,
            distill=config.distill,
            num_group=config.num_group,
            total_instance=config.total_instance,
            bag_weight=config.bag_weight,
            num_residuals=config.num_residuals
        )
        self.forward = self.model.forward
        self.forward_attention = self.model.forward_attention
        self.forward_features = self.model.forward_features
        self.forward_head = self.model.forward_head
        self.initialize_classifier = self.model.initialize_classifier


AutoConfig.register("dftd", DFTDConfig)
AutoModel.register(DFTDConfig, DFTDModel)

