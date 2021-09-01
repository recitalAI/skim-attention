from typing import List, Union
from transformers.utils import logging
from .configuration_skim import SkimformerConfig

logger = logging.get_logger(__name__)

class LongSkimformerConfig(SkimformerConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LongformerModel` or a
    :class:`~transformers.TFLongformerModel`. It is used to instantiate a Longformer model according to the specified
    arguments, defining the model architecture.

    This is the configuration class to store the configuration of a :class:`~transformers.LongformerModel`. It is used
    to instantiate an Longformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    `roberta-base <https://huggingface.co/roberta-base>`__ architecture with a sequence length 4,096.

    The :class:`~transformers.LongformerConfig` class directly inherits :class:`~transformers.RobertaConfig`. It reuses
    the same defaults. Please check the parent class for more information.

    Args:
        attention_window (:obj:`int` or :obj:`List[int]`, `optional`, defaults to 512):
            Size of an attention window around each token. If an :obj:`int`, use the same size for all layers. To
            specify a different window size for each layer, use a :obj:`List[int]` where ``len(attention_window) ==
            num_hidden_layers``.

    Example::

        >>> from transformers import LongformerConfig, LongformerModel

        >>> # Initializing a Longformer configuration
        >>> configuration = LongformerConfig()

        >>> # Initializing a model from the configuration
        >>> model = LongformerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "longskimformer"

    def __init__(
        self, 
        max_position_embeddings: int = 4098,
        attention_window: Union[List[int], int] = 512, 
        pad_token_id: int = 1,
        sep_token_id: int = 2, 
        type_vocab_size: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        ignore_attention_mask: bool = False,
        layer_norm_eps: float = 1e-05,
        pad_token_bbox_value: int = 0,
        **kwargs
    ):
        super().__init__(
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id, 
            sep_token_id=sep_token_id, 
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            **kwargs
        )
        self.attention_window = attention_window
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.ignore_attention_mask = ignore_attention_mask
        self.pad_token_bbox_value = pad_token_bbox_value
