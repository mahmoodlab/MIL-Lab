from loguru import logger
import pdb
import torch.nn as nn
import yaml
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from src.builders.ModelDict import ModelDict
from src.builder_utils import _cfg, build_model_with_cfg
from src._global_mappings import REPO_PATH, CONFIG_PATH, ENCODER_DIM_MAPPING, MODEL_ENTRYPOINTS, MODEL_SAVE_PATH  # todo, should this be all caps or not?
from pathlib import Path

def save_model(model: nn.Module,
               model_name: str,
               save_folder: str = MODEL_SAVE_PATH,
               save_pretrained: bool = False):
    """
    Save a model to the specified path
    Args:
        model (nn.Module): The model to save.
        model_name (str): The name of the model, used to create a subdirectory.
        save_folder (str): The directory where the model will be saved.
        save_pretrained (bool): If True, use the Hugging Face save_pretrained method if available.
    """
    from transformers import PreTrainedModel
    import torch
    model_path = os.path.join(save_folder, model_name)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    if isinstance(model, PreTrainedModel) and save_pretrained:
        model.save_pretrained(model_path)
        if hasattr(model, 'config'):
            model.config.save_pretrained(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
    print(f'Model saved to {model_path}')


def create_model(
        model_name: str,
        num_classes: int = 2,
        checkpoint_path: str = '',
        hf_base_repo: str = 'mahmoodlab/',
        from_pretrained: bool = False,
        pretrained_strict: bool = False,
        keep_classifier: bool = False,
        **kwargs,
):
    """
    Create and return a model instance given a model name and configuration options.

    Args:
        model_name (str): The full model name string, e.g. "abmil.base.uni.op-108".
        num_classes (int): Number of output classes for the model. Defaults to -1.
        checkpoint_path (str, optional): Path to a checkpoint file to load weights from. If provided, overrides local path in pretrained_cfg. Defaults to ''.
        hf_base_repo (str, optional): Base Hugging Face repository for model weights. Defaults to 'mahmoodlab/'.
        from_pretrained (bool, optional): If True, load model using Hugging Face's from_pretrained method. Defaults to False.
        pretrained_strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the model. Defaults to False.
        keep_classifier (bool, optional): Whether to keep or remove classification head from pretrained model. Defaults to False.
        **kwargs: Additional keyword arguments passed to the model builder.

    Returns:
        nn.Module: Instantiated model object.

    Raises:
        ValueError: If pretrained=True but no pretrained config is available for the model.
        NotImplementedError: If the model name is not in the available models.
    """
    model_dict = ModelDict.from_string(model_name)
    pretrained_cfg = _create_pretrained_config(model_dict,
                                               hf_source=hf_base_repo,
                                               local_source=MODEL_SAVE_PATH)

    # override checkpoint path if provided
    pretrained_cfg = _update_checkpoint_path(checkpoint_path, pretrained_cfg)
    from_pretrained = from_pretrained and model_dict.is_pretrained()
    model = build_model(
        model_name=model_dict.model_name,
        model_config=model_dict.model_config,
        pretrained=model_dict.is_pretrained(),
        encoder=model_dict.encoder,
        num_classes=num_classes,
        pretrained_cfg=pretrained_cfg,
        from_pretrained=from_pretrained ,
        pretrained_strict=pretrained_strict,
        keep_classifier=keep_classifier,
        **kwargs,
    )

    
    return model

def _update_checkpoint_path(checkpoint_path: str, pretrained_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the checkpoint path in the pretrained config.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):  
        logger.warning(f'Checkpoint path manually provided ({checkpoint_path}). Overwriting previous local filepath {pretrained_cfg.get("local_path", "")}.')
        pretrained_cfg['file'] = checkpoint_path
    return pretrained_cfg


def build_model(
    model_name: str,
    model_config: str,
    pretrained: bool,
    encoder: str,
    num_classes: int,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    from_pretrained: bool = False,
    pretrained_strict: bool = False,
    keep_classifier: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Build and instantiate a model given its name, config, encoder, and configuration.

    Args:
        model_name (str): Name of the model architecture (e.g., 'transmil', 'abmil').
        model_config (str): config of the model (e.g., 'base', 'large').
        pretrained (bool): whether to load pretrained model
        encoder (str): Encoder type to use (e.g., 'uni', 'resnet').
        num_classes (int): Number of output classes for the classification head.
        pretrained_cfg (Optional[Dict[str, Any]], optional): Pretrained configuration dictionary. Defaults to None.
        from_pretrained (bool, optional): If True, use Hugging Face's from_pretrained method. Defaults to False.
        pretrained_strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the model. Defaults to False.
        keep_classifier (bool, optional): Whether to keep or remove classification head from pretrained model. Defaults to True.
        **kwargs: Additional keyword arguments passed to the model builder.

    Returns:
        nn.Module: Instantiated model object.

    Raises:
        KeyError: If the model_name is not found in MODEL_ENTRYPOINTS.
        FileNotFoundError: If the configuration file for the model cannot be found.
        Exception: For other errors during model instantiation.
    """
    config = _load_model_config(model_name, model_config)
    config['in_dim'] = ENCODER_DIM_MAPPING[encoder]
    config['num_classes'] = num_classes

    model_cls, config_cls = MODEL_ENTRYPOINTS[model_name]
    model = build_model_with_cfg(
        model_cls, num_classes=num_classes,
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        model_cfg=config_cls(**config),
        from_pretrained=from_pretrained,
        pretrained_strict=pretrained_strict,
        keep_classifier=keep_classifier,
        **kwargs)
    return model


def _load_model_config(model_name: str, model_config: str) -> Dict:
    """
    Load the configuration dictionary for a given model name and config.

    Args:
        model_name (str): The name of the model architecture (e.g., 'transmil', 'abmil').
        model_config (str): The config of the model (e.g., 'base', 'default').

    Returns:
        Dict: The configuration dictionary loaded from the corresponding YAML file.
    """
    config_path = os.path.join(CONFIG_PATH, model_name, f'{model_config}.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    return config


def _create_pretrained_config(name_dict: ModelDict,
                   hf_source: str = 'MahmoodLab/',
                   local_source: str = 'model_weights'
) -> Dict:
    """
    Generate a default configuration dictionary for a model.

    Args:
        name_dict (ModelDict): An object representing the parsed model name and its components.
        hf_source (str, optional): Base Hugging Face repository or user/org name. Defaults to 'mahmoodlab/'.
        local_source (str, optional): Local directory name where model weights are stored. Defaults to 'model_weights'.

    Returns:
        Dict: A configuration dictionary containing keys such as 'hf_hub_id', 'local_path_parent', and 'local_path'.
    """
    default_cfg = _cfg() 
    model_path = name_dict.to_string()
    default_cfg['hf_hub_id'] = os.path.join(hf_source, model_path).replace('\\', '/')
    default_cfg['local_path_parent'] = os.path.join(REPO_PATH, local_source, model_path)
    default_cfg['local_path'] = os.path.join(REPO_PATH, local_source, model_path)
    return default_cfg

