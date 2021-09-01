from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import BertConfig

logger = logging.get_logger(__name__)

class SkimformerConfig(PretrainedConfig):
    model_type = "skimformer"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        hidden_layout_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        attention_head_size=64, 
        skim_attention_head_size=64,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_2d_position_embeddings=1024,
        use_1d_positions=False,
        degrade_2d_positions=False,
        contextualize_2d_positions=False,
        num_hidden_layers_layout_encoder=2,
        num_attention_heads_layout_encoder=12,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_layout_size = hidden_layout_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.skim_attention_head_size = skim_attention_head_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.use_1d_positions = use_1d_positions
        self.degrade_2d_positions = degrade_2d_positions
        self.contextualize_2d_positions = contextualize_2d_positions
        self.num_hidden_layers_layout_encoder = num_hidden_layers_layout_encoder
        self.num_attention_heads_layout_encoder = num_attention_heads_layout_encoder
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing


class BertWithSkimEmbedConfig(BertConfig):
    model_type = "bertwithskimembed"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        hidden_layout_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        max_2d_position_embeddings=1024,
        num_hidden_layers_layout_encoder=2,
        num_attention_heads_layout_encoder=12,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.hidden_layout_size = hidden_layout_size
        self.num_hidden_layers_layout_encoder = num_hidden_layers_layout_encoder
        self.num_attention_heads_layout_encoder = num_attention_heads_layout_encoder

class SkimmingMaskConfig(BertConfig):
    model_type = "skimmingmask"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        hidden_layout_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_skim_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        skim_attention_head_size=64,
        pad_token_id=0,
        gradient_checkpointing=False,
        max_2d_position_embeddings=1024,
        contextualize_2d_positions=False,
        num_hidden_layers_layout_encoder=2,
        num_attention_heads_layout_encoder=12,
        top_k=0,
        core_model_type="bert",
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.hidden_layout_size = hidden_layout_size
        self.num_skim_attention_heads = num_skim_attention_heads
        self.skim_attention_head_size = skim_attention_head_size
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.contextualize_2d_positions = contextualize_2d_positions
        self.num_hidden_layers_layout_encoder = num_hidden_layers_layout_encoder
        self.num_attention_heads_layout_encoder = num_attention_heads_layout_encoder

        self.top_k = top_k if top_k else 0
        self.core_model_type = core_model_type