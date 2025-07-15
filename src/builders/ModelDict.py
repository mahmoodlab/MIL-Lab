from dataclasses import dataclass

from typing import Optional, Dict, Optional
import warnings

NO_PRETRAIN_STR = 'none'
DEFAULT_ENCODER = 'uni'
DEFAULT_CONFIG = 'base'

@dataclass
class ModelDict:
    """
    A dataclass representing a model configuration with fields for model name, config, encoder, and task.

    Attributes:
        model_name (str): The name of the model architecture (e.g., 'abmil').
        model_config (str): The config of the model (e.g., 'base', 'small').
        encoder (str): The encoder type used by the model (e.g., 'uni', 'resnet').
        task (str): The task or dataset identifier (e.g., 'ot108', 'none').
    """

    model_name: str
    model_config: str
    encoder: str
    task: str

    def __post_init__(self):
        """
        Validate the model name format after initialization.

        Raises:
            ValueError: If any of the fields are empty.
        """
        if not all([self.model_name, self.model_config, self.encoder, self.task]):
            raise ValueError("All fields must be non-empty strings")



    @staticmethod
    def _infer_random_task(model_name: str) -> str:
        """
        Check if only 3 parts are provided, and if so, infer the task as 'none'
        """
        if len(model_name.split('.')) == 3:
            print(f'"Model name {model_name} does not have a task, using default task {NO_PRETRAIN_STR}.')
            return f'{model_name}.{NO_PRETRAIN_STR}'
        return model_name

    @staticmethod
    def _infer_encoder(model_name: str) -> str:
        """
        Infer the encoder type from the model name if not explicitly provided.

        Args:
            model_name (str): The model name string.

        Returns:
            str: The inferred encoder type.
        """
        parts = model_name.split('.')
        if len(parts) == 2:
            # If only model_name and model_config are provided, use the default encoder
            print(f'"Model name {model_name} does not have an encoder, using default encoder {DEFAULT_ENCODER}.')
            return f"{parts[0]}.{parts[1]}.{DEFAULT_ENCODER}"
        return model_name


    @staticmethod
    def _infer_config(model_name: str):
        parts = model_name.split('.')
        if len(parts) == 1:
            print(f'"Model name {model_name} does not have a config, using default config {DEFAULT_CONFIG}.')
            return f'{model_name}.{DEFAULT_CONFIG}'
        return model_name


    @staticmethod
    def format_string(model_string: str):
        """
        Format the model string to ensure it has exactly 4 parts. Infer the encoder and task if necessary.

        Args:
            model_string (str): A string in the format 'model_name.model_config.encoder.task'.

        Returns:
            str: The formatted model string with exactly 4 parts.
        """
        model_string = ModelDict._infer_config(model_string)
        model_string = ModelDict._infer_encoder(model_string)
        model_string = ModelDict._infer_random_task(model_string)

        assert model_string.count('.') == 3, f"Model string must have exactly 3 dots for modeltype.config.encoder.task, \
                                                    got {model_string.count('.')} dots"
        return model_string


    @staticmethod
    def from_string(model_string: str, pretrained: bool = True) -> 'ModelDict':
        """
        Create a ModelDict instance from a dot-separated string.

        Args:
            model_string (str): A string in the format 'model_name.model_config.encoder.task'.
            pretrained (bool): If False, then the task is set to 'none'. If True, the task is inferred from the string.

        Returns:
            ModelDict: An instance of ModelDict parsed from the string.

        Raises:
            ValueError: If the string does not have exactly 4 parts.
        """
        model_string = ModelDict.format_string(model_string)

        parts = model_string.split('.')
        if len(parts) != 4:
            raise ValueError(f"Model string must have exactly 4 parts separated by dots, got {len(parts)} parts")

        config = ModelDict(
            model_name=parts[0],
            model_config=parts[1],
            encoder=parts[2],
            task=parts[3]
        )

        config.check_pretrained_flag(pretrained=pretrained)
        return config


    @staticmethod
    def from_dict(model_dict: Dict[str, str]) -> 'ModelDict':
        """
        Create a ModelDict instance from a dictionary.

        Args:
            model_dict (Dict[str, str]): A dictionary with keys 'model_name', 'model_config', 'encoder', and 'task'.

        Returns:
            ModelDict: An instance of ModelDict created from the dictionary.

        Raises:
            ValueError: If any required key is missing.
            TypeError: If any value is not a string.
        """
        required_keys = ['model_name', 'model_config', 'encoder']
        for key in required_keys:
            if key not in model_dict:
                raise ValueError(f"Missing required key '{key}' in model_dict")
            if not isinstance(model_dict[key], str):
                raise TypeError(
                    f"Value for key '{key}' must be a string, got {type(model_dict[key])}"
                )
        return ModelDict(
            model_name=model_dict['model_name'],
            model_config=model_dict['model_config'],
            encoder=model_dict['encoder'],
            task=model_dict.get('task', NO_PRETRAIN_STR)
        )

    def is_pretrained(self) -> bool:
        """
        Determine if the model is pretrained based on the task field.

        Returns:
            bool: True if the task is not 'none', indicating pretrained; False otherwise.
        """
        return self.task != NO_PRETRAIN_STR

    def to_string(self) -> str:
        """
        Convert the ModelDict instance back to a dot-separated string.

        Returns:
            str: The dot-separated string representation.
        """
        return f"{self.model_name}.{self.model_config}.{self.encoder}.{self.task}"


    def check_pretrained_flag(self, pretrained: bool):
        """
        Check if the pretrained flag is set correctly based on the task.

        Args:
            model_str (bool): The model string to check.
            pretrained (bool): If False, then the task is set to 'none'. If True, the task is inferred from the string.

        Returns:
            str: The task string based on the pretrained flag.
        """
        if pretrained:
            if self.is_pretrained():
                warnings.warn("Pretrained flag is True, but task is set to 'none'. Using random weights")
        else:
            if not self.is_pretrained():
                warnings.warn(f'Pretrained flag is False, updating pretrain task from '
                              f'"{self.task}" to "{NO_PRETRAIN_STR}".')
                self.task = NO_PRETRAIN_STR


    def __eq__(self, other: 'ModelDict') -> bool:
        """
        Check equality with another ModelDict instance.

        Args:
            other (ModelDict): Another ModelDict instance.

        Returns:
            bool: True if both instances have the same string representation, False otherwise.
        """
        return self.to_string() == other.to_string()

    def __hash__(self) -> int:
        """
        Compute the hash value of the ModelDict instance.

        Returns:
            int: The hash of the string representation.
        """
        return hash(self.to_string())

    def __str__(self) -> str:
        """
        Return the string representation of the ModelDict instance.

        Returns:
            str: The dot-separated string representation.
        """
        return self.to_string()

    def __repr__(self) -> str:
        """
        Return the official string representation of the ModelDict instance.

        Returns:
            str: The string representation in the form 'ModelDict(...)'.
        """
        return f"ModelDict({self.to_string()})"
