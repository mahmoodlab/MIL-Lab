import pdb
from pathlib import Path
import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoConfig
from loguru import logger
from torch import nn
try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

from torch.hub import load_state_dict_from_url
### helper tools for builder.py

_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False

def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'head': 'head',
        **kwargs,
    }


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans a state dictionary by removing the .module prefix from keys if it exists.

    Args:
        state_dict (Dict[str, Any]): The original state dictionary whose keys will be cleaned.

    Returns:
        Dict[str, Any]: A new state dictionary with the .module prefix removed from keys.
    """
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def _resolve_pretrained_source(pretrained_cfg: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Decide how to load pretrained weights from pretrained_cfg.

    The function determines the source of pretrained weights based on the provided configuration dictionary.
    The priority order for loading is:
        1. If 'source' is specified in the config, use the corresponding entry in the config.
        2. If a local file is specified and exists, use the local file.
        3. If a Hugging Face Hub ID is specified, use the hub.
        4. If a URL is specified, use the URL.
        5. Otherwise, raise an error.

    Args:
        pretrained_cfg (Dict[str, Any]): Configuration dictionary containing possible keys:
            - 'source': (str) Key to use as the source in the config.
            - 'url': (str) URL to download weights from.
            - 'file': (str) Local file path to load weights from.
            - 'hf_hub_id': (str) Hugging Face Hub model ID.

    Returns:
        Tuple[str, Any]: A tuple of (source_type, source_value), where source_type is one of
            'file', 'hf-hub', 'url', or the value of 'source', and source_value is the corresponding value.

    Raises:
        ValueError: If no valid pretrained weights source is found in the config.
    """
    cfg_source = pretrained_cfg.get('source', '')
    pretrained_url = pretrained_cfg.get('url', None)
    pretrained_file = pretrained_cfg.get('file', None)
    hf_hub_id = pretrained_cfg.get('hf_hub_id', None)

    # resolve where to load pretrained weights from
    if cfg_source:
        assert cfg_source in pretrained_cfg, f"Source {cfg_source} not found in pretrained_cfg"
        return cfg_source, pretrained_cfg[cfg_source]
    
    if pretrained_file and os.path.exists(pretrained_file):
        return 'file', pretrained_file
    elif hf_hub_id:
        return 'hf-hub', hf_hub_id
    elif pretrained_url:
        return 'url', pretrained_url
    raise ValueError(f"No pretrained weights found in {pretrained_cfg}")

def load_state_dict(
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
) -> Dict[str, Any]:
    """
    Loads a state dictionary from a checkpoint file.

    This function supports loading from both standard PyTorch checkpoint files and safetensors files.
    It will attempt to extract the most appropriate state dict from the checkpoint, preferring
    Exponential Moving Average (EMA) weights if available and requested.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        use_ema (bool, optional): If True, prefer loading EMA weights if present. Defaults to True.
        device (Union[str, torch.device], optional): Device to map the loaded tensors to. Defaults to 'cpu'.

    Returns:
        Dict[str, Any]: The loaded state dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        AssertionError: If loading a .safetensors file but the safetensors package is not installed.
    """

    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        logger.info("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def _find_pretrained_source(hf_path, local_folder):
    path_to_local = os.path.join(local_folder, hf_path.split('/')[-1])
    weights_path = os.path.join(path_to_local, 'model.safetensors')
    config_path = os.path.join(path_to_local, 'config.json')
    if os.path.exists(weights_path) and os.path.exists(config_path):
        return path_to_local
    hf_path, subfolder = _clean_model_id(hf_path, '')
    assert subfolder == '', f'loading from_pretrained does not allow subfolders in repo: received {hf_path}/{subfolder}'

    from huggingface_hub import snapshot_download
    hf_path = snapshot_download(
        repo_id=hf_path,
        revision="main",
        allow_patterns=["*.py", "model.safetensors", "pytorch_model.bin", "config.json"]
    )
    return hf_path


def load_from_hf(hf_path: str, local_folder: str) -> nn.Module:
    """
    Loads a model from the Hugging Face Hub using the given model path.

    Args:
        hf_path (str): The Hugging Face model identifier, which may include subfolders.

    Returns:
        nn.Module: The loaded model as a PyTorch module.

    Notes:
        - This function uses transformers.AutoModel to load the model.
        - If the model_id contains subfolders, they are handled appropriately.
        - trust_remote_code=True is used to allow loading custom model code from the repository.
    """
    from transformers import AutoModel
    model_path = _find_pretrained_source(hf_path, local_folder)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    return model

def _clean_model_id(model_id: str, filename: str) -> Tuple[str, str]:
    """
    Cleans a Hugging Face model_id to ensure it contains at most one slash (repo_id/model_name).
    If the model_id contains more than one slash, the first two parts are used as the model_id,
    and the remaining parts are joined (with the provided filename) to form a subfolder or file path.

    Args:
        model_id (str): The Hugging Face model identifier, possibly with subfolders (e.g., 'user/repo/subfolder1/subfolder2').
        filename (str): The filename or subfolder to append.

    Returns:
        Tuple[str, str]: A tuple containing the cleaned model_id and the updated filename/subfolder.
    """
    if model_id.count("/") > 1:  # repo_id can only have 1 slash or HF will get mad
        parts = model_id.split('/')
        model_id = '/'.join(parts[:2])  # extract first two parts
        filename = os.path.join(*parts[2:], filename)  # join the rest as filename
    return model_id, filename


def _append_prefix_to_state_dict(
    state_dict: Dict[str, Any],
    prefix: str
) -> Dict[str, Any]:
    """
    Appends a prefix to each key in the state dictionary if it does not already start with the prefix.

    Args:
        state_dict (Dict[str, Any]): The original state dictionary whose keys will be prefixed.
        prefix (str): The prefix to prepend to each key.

    Returns:
        Dict[str, Any]: A new state dictionary with updated keys.
    """
    new_state_dict: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            new_state_dict[f'{prefix}.{k}'] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

from typing import Dict

def download_hf_state_dict(
    model_id: str = 'bert-base-uncased',
    save_path: str = '../model_weights',
    filename: str = 'pytorch_model.bin'
) -> Dict:
    """
    Download a state dictionary (.bin file) from the Hugging Face Hub and return it as a PyTorch state dict.

    Args:
        model_id (str): The Hugging Face model repository ID (e.g., 'bert-base-uncased' or 'user/model').
        save_path (str): Local directory to save the downloaded file.
        filename (str): Name of the file to download (usually 'pytorch_model.bin').

    Returns:
        Dict: The loaded PyTorch state dictionary.

    Raises:
        SystemExit: If the download or loading fails.
    """
    from huggingface_hub import hf_hub_download
    import torch
    from pathlib import Path
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # --- Download the .bin file ---
    try:
        print(f"Attempting to download '{filename}' from '{model_id}'...")
        # hf_hub_download downloads the file to your local Hugging Face cache
        # It returns the local path to the downloaded file
        repo_id, filename = _clean_model_id(model_id, filename)
        local_path = hf_hub_download(repo_id=repo_id,
                                     filename=filename,
                                     local_dir=save_path,
                                     revision='main')

        print(f"State dict downloaded successfully to: {local_path}")

        state_dict = torch.load(local_path, map_location='cpu')  
        state_dict = _append_prefix_to_state_dict(state_dict, 'model.')

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Could not download or load '{filename}' from '{model_id}'.")
        print("Please check the 'model_id' and 'filename' to ensure they are correct and the file exists.")
        print("You can verify files on the Hugging Face model page under the 'Files' tab.")
        import sys
        sys.exit()
    return state_dict


def load_pretrained(
        model: nn.Module,
        num_classes: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
        from_pretrained: bool = False,
        keep_classifier: bool = False,
) -> nn.Module:
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        num_classes (int): num_classes for target model
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        in_chans (int): in_chans for target model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        from_pretrained (bool): whether to load from pretrained model or initialize our own model
        keep_classifier (bool): whether to keep or remove classification head from pretrained model

    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg :
        raise RuntimeError("Invalid pretrained config, cannot load weights. Use `pretrained=False` for random init.")

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)

    if from_pretrained:
        model = load_from_hf(pretrained_loc, local_folder=pretrained_cfg['local_path_parent'])
        state_dict = model.state_dict()
    elif load_from == 'hf-hub':
        logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        state_dict = download_hf_state_dict(pretrained_loc, save_path=pretrained_cfg['local_path_parent'])

    elif load_from == 'file':
        logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        state_dict = load_state_dict_from_url(
            pretrained_loc,
            map_location='cpu',
            progress=_DOWNLOAD_PROGRESS,
            check_hash=_CHECK_HASH,
            strict=strict,
        )
    else:
        model_name = pretrained_cfg.get('architecture', 'this model')
        raise RuntimeError(f"No pretrained weights exist for {model_name}. Use `pretrained=False` for random init.")

    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError as e:
            # for backwards compat with filter fn that take one arg
            state_dict = filter_fn(state_dict)
    if not keep_classifier:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.classifier.')}
    if hasattr(model, 'clean_state_dict'):
        state_dict = model.clean_state_dict(state_dict)
    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys:
        logger.info(
            f'Missing keys ({", ".join(load_result.missing_keys)}) discovered while loading pretrained weights.'
            f' This is expected if model is being adapted.')
    if load_result.unexpected_keys:
        logger.warning(
            f'Unexpected keys ({", ".join(load_result.unexpected_keys)}) found while loading pretrained weights.'
            f' This may be expected if model is being adapted.')
    if not keep_classifier:
        print(f'Removing classifier from pretrained model')
        model.model.initialize_classifier(num_classes=num_classes)
    return model


def build_model_with_cfg(
        model_cls: Callable,
        num_classes: int,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        from_pretrained: bool = True,
        keep_classifier: bool = False,
        **kwargs,
):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls: model class
        num_classes: number of classes for classification head
        pretrained: whethe to load pretrained model
        pretrained_cfg: model's pretrained weight/task config
        model_cfg: model's architecture config
        pretrained_strict: load pretrained weights strictly
        pretrained_filter_fn: filter callable for pretrained weights
        from_pretrained: whether to load from huggingface AutoModel or from state dict
        keep_classifier: whether to keep or remove classification head from pretrained model
        kwargs_filter: kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """

    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(config=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    if pretrained:
        model = load_pretrained(
            model,
            num_classes=num_classes,
            pretrained_cfg=pretrained_cfg,
            filter_fn=pretrained_filter_fn,
            strict=pretrained_strict,
            from_pretrained=from_pretrained,
            keep_classifier=keep_classifier,
            )

    return model

