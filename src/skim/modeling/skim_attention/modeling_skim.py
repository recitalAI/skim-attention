from collections import namedtuple
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm

from torch.autograd.function import Function

from transformers.file_utils import (
    ModelOutput,
)

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    TokenClassifierOutput,
)

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertEncoder,
    BertOnlyMLMHead,
)

from transformers.models.layoutlm.modeling_layoutlm import LayoutLMEmbeddings

from .configuration_skim import (
    SkimformerConfig,
    BertWithSkimEmbedConfig,
    SkimmingMaskConfig,
)

logger = logging.getLogger(__name__)

SkimformerEncoderOutput = namedtuple(
    "SkimformerEncoderOutput",
    ["hidden_states", "all_hidden_states"],
)

class SkimformerTextEmbeddings(nn.Module):
    """Construct the text embeddings from word and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Skimformer1DPositionEmbeddings(nn.Module):
    """Construct sequential position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_layout_size)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.LayerNorm = LayerNorm(config.hidden_layout_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_shape, device, position_ids=None):
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                0,  seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        position_embeddings = self.position_embeddings(position_ids)
        
        position_embeddings = self.LayerNorm(position_embeddings)
        position_embeddings = self.dropout(position_embeddings)
        return position_embeddings
        
class Skimformer2DPositionEmbeddings(nn.Module):
    """Construct the layout embeddings from the bounding box coordinates."""
    def __init__(self, config):
        super().__init__()
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)
        self.degrade_2d_positions = config.degrade_2d_positions if hasattr(config, "degrade_2d_positions") else False

        self.LayerNorm = LayerNorm(config.hidden_layout_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, bbox=None):
        if self.degrade_2d_positions:
            try:
                x_center = (bbox[:, :, 0] + bbox[:, :, 2]) // 2
                y_center = (bbox[:, :, 1] + bbox[:, :, 3]) // 2
                x_center_position_embeddings = self.x_position_embeddings(x_center)
                y_center_position_embeddings = self.y_position_embeddings(y_center)
            except IndexError as e:
                raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.") from e

            embeddings = x_center_position_embeddings + y_center_position_embeddings
        else:
            try:
                left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
                upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
                right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
                lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
            except IndexError as e:
                raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.") from e

            h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
            w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

            embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
                + h_position_embeddings
                + w_position_embeddings
            )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertWithSkimEmbedEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_layout_size)

        self.x_position_projection = nn.Linear(config.hidden_layout_size, config.hidden_size)
        self.y_position_projection = nn.Linear(config.hidden_layout_size, config.hidden_size)
        self.h_position_projection = nn.Linear(config.hidden_layout_size, config.hidden_size)
        self.w_position_projection = nn.Linear(config.hidden_layout_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        # project into same dimension as text embeddings
        left_position_embeddings = self.x_position_projection(left_position_embeddings)
        upper_position_embeddings = self.y_position_projection(upper_position_embeddings)
        right_position_embeddings = self.x_position_projection(right_position_embeddings)
        lower_position_embeddings = self.y_position_projection(lower_position_embeddings)

        h_position_embeddings = self.h_position_projection(h_position_embeddings)
        w_position_embeddings = self.w_position_projection(w_position_embeddings)

        two_dim_pos_embeddings = (
            left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
        )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            + position_embeddings
            + two_dim_pos_embeddings
            + token_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SkimAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.skim_attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_layout_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_layout_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_layout_states,
        attention_mask=None,
    ):

        key_layer = self.transpose_for_scores(self.key(hidden_layout_states))
        query_layer = self.transpose_for_scores(self.query(hidden_layout_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # return the attention probabilities only: Softmax(QK^T/sqrt(d))
        return attention_probs  

class SkimformerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_probs,
        head_mask=None,
    ):
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Softmax(QK^T/sqrt(d)) . V
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class SkimformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dense = nn.Linear(all_head_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SkimformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SkimformerSelfAttention(config)
        self.output = SkimformerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)


    def forward(
        self,
        hidden_states,
        attention_probs,
        head_mask=None,
    ):
        if len(self.pruned_heads) > 0:
            num_attention_heads = attention_probs.shape[1]
            indices = [idx for idx in range(num_attention_heads) if idx not in self.pruned_heads]
            attention_probs = torch.index_select(attention_probs, 1, indices)

        self_output = self.self(
            hidden_states,
            attention_probs,
            head_mask,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class SkimformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SkimformerAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_probs,
        head_mask=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_probs,
            head_mask,
        )

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class SkimformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SkimformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_probs,
        head_mask=None,
        output_hidden_states=False,
        return_dict=None,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_probs,
                    layer_head_mask,
                )
            else:
                layer_output = layer_module(
                    hidden_states,
                    attention_probs,
                    layer_head_mask,
                )

            hidden_states = layer_output

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                ]
                if v is not None
            )
        return SkimformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
        )


class SkimformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SkimformerConfig
    base_model_prefix = "skimformer"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertWithSkimEmbedPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertWithSkimEmbedConfig
    base_model_prefix = "bertwithskimembed"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class SkimmingMaskPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SkimmingMaskConfig
    base_model_prefix = "skimmingmask"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@dataclass
class SkimformerModelOutput(ModelOutput):
    """
    Output type of :class:`~SkimformerModel`.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    attentions: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    

class SkimformerModel(SkimformerPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.use_1d_positions = config.use_1d_positions

        self.text_embeddings = SkimformerTextEmbeddings(config)
        
        if self.use_1d_positions:
            self.one_dim_pos_embeddings = Skimformer1DPositionEmbeddings(config)
        else:
            self.two_dim_pos_embeddings = Skimformer2DPositionEmbeddings(config)

        self.contextualize_2d_positions = config.contextualize_2d_positions
        if self.contextualize_2d_positions:
            self.layout_encoder = BertEncoder(
                BertConfig(
                    hidden_size=config.hidden_layout_size,
                    num_hidden_layers=config.num_hidden_layers_layout_encoder,
                    num_attention_heads=config.num_attention_heads_layout_encoder,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    max_position_embeddings=config.max_2d_position_embeddings,
                    initializer_range=config.initializer_range,
                    layer_norm_eps=config.layer_norm_eps,
                    gradient_checkpointing=config.gradient_checkpointing,
                )
            )
        self.skim_attention = SkimAttention(config)
        self.encoder = SkimformerEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert (
            len(input_shape) == 2
        ), "`input_ids` has to be of shape `[batch_size, sequence_length]`, but got shape: {}".format(input_shape)

        if bbox is not None:
            bbox_shape = bbox.size()
            assert (
                len(bbox_shape) == 3 
            ), "`bbox` has to be of shape `[batch_size, sequence_length, 4]`, but got shape: {}".format(bbox_shape)

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        if self.use_1d_positions:
            pos_embedding_output = self.one_dim_pos_embeddings(
                input_shape=input_shape,
                device=device,
                position_ids=position_ids,
            )
        else:
            pos_embedding_output = self.two_dim_pos_embeddings(
                bbox=bbox,
            )
            if self.contextualize_2d_positions:
                pos_embedding_output = self.layout_encoder(
                    hidden_states=pos_embedding_output,
                )[0]

        skim_attention_output = self.skim_attention(
            pos_embedding_output,
            attention_mask=extended_attention_mask,
        )

        encoder_outputs = self.encoder(
            text_embedding_output,
            skim_attention_output,
            head_mask=head_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            outputs = (sequence_output, pooled_output)
            if output_attentions:
                outputs = outputs + (skim_attention_output, )
            if output_hidden_states:
                outputs = outputs + encoder_outputs[1:]
            return outputs

        return SkimformerModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            attentions=skim_attention_output if output_attentions else None,
            hidden_states=encoder_outputs.all_hidden_states,
        )

class BertWithSkimEmbedModel(BertWithSkimEmbedPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertWithSkimEmbedEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class SkimmingMaskModel(SkimmingMaskPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config 

        self.core_model_type = config.core_model_type 
        self.embeddings = BertEmbeddings(config) if self.core_model_type == "bert" else LayoutLMEmbeddings(config)
        self.two_dim_pos_embeddings = Skimformer2DPositionEmbeddings(config)
        self.contextualize_2d_positions = config.contextualize_2d_positions
        if config.contextualize_2d_positions:
            self.layout_encoder = BertEncoder(
                BertConfig(
                    hidden_size=config.hidden_layout_size,
                    num_hidden_layers=config.num_hidden_layers_layout_encoder,
                    num_attention_heads=config.num_attention_heads_layout_encoder,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    max_position_embeddings=config.max_2d_position_embeddings,
                    initializer_range=config.initializer_range,
                    layer_norm_eps=config.layer_norm_eps,
                    gradient_checkpointing=config.gradient_checkpointing,
                )
            )

        self.skim_attention = SkimAttention(config) 
        self.top_k = config.top_k 

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if self.core_model_type == "bert":
            text_embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
        else:
            text_embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
        spatial_pos_embedding_output = self.two_dim_pos_embeddings(bbox=bbox)

        if self.contextualize_2d_positions:
            spatial_pos_embedding_output = self.layout_encoder(hidden_states=spatial_pos_embedding_output)[0]

        skim_attention_output = self.skim_attention(
            spatial_pos_embedding_output,
            attention_mask=extended_attention_mask,
        )

        topk_idx = torch.topk(skim_attention_output, self.top_k, -1).indices
        skim_attention_mask = torch.zeros(skim_attention_output.shape, device=device)
        skim_attention_mask = skim_attention_mask.scatter(-1, topk_idx, 1)
        skim_attention_mask = skim_attention_mask * attention_mask[:, None, :, :]

        skim_attention_mask = (1.0 - skim_attention_mask) * -10000.0           

        encoder_outputs = self.encoder(
            text_embedding_output,
            skim_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class SkimformerForMaskedLM(SkimformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.skimformer = SkimformerModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.skimformer(
            input_ids,
            bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class SkimformerForTokenClassification(SkimformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.skimformer = SkimformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.skimformer(
            input_ids,
            bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertWithSkimEmbedForTokenClassification(BertWithSkimEmbedPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert_with_skim_embed = BertWithSkimEmbedModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.bert_with_skim_embed.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert_with_skim_embed.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert_with_skim_embed(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class SkimmingMaskForTokenClassification(SkimmingMaskPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.skimming_mask_model = SkimmingMaskModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.skimming_mask_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.skimming_mask_model.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.skimming_mask_model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                if attention_mask.dim() == 3:
                    active_loss = (torch.sum(attention_mask, dim=-1)).view(-1) > 0
                else:
                    active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )