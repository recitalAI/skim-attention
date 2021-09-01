from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from transformers.tokenization_utils_base import PaddingStrategy, BatchEncoding, PreTrainedTokenizerBase


def _collate_batch(
    examples, 
    tokenizer, 
    pad_to_multiple_of: Optional[int] = None,
    pad_token_bbox_value: int = 0,
):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    bboxes = None
    
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [
            torch.tensor(e, dtype=torch.long) for e in examples
        ]
    elif isinstance(
        examples[0], 
        Union[Tuple[List[int], List[int]]]
    ):
        examples = [torch.tensor(e[0], dtype=torch.long) for e in examples]
        bboxes = [torch.tensor(e[1], dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        output = (torch.stack(examples, dim=0), )
        if bboxes:
            output = output + (torch.stack(bboxes, dim=0), )
        return output

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    if bboxes:
        bbox_result = bboxes[0].new_full([len(examples), max_length, 4], pad_token_bbox_value)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
            if bboxes:
                bbox_result[i, : bboxes[i].shape[0]] = bboxes[i]
        else:
            result[i, -example.shape[0] :] = example
            if bboxes:
                bbox_result[i, -bboxes.shape[0] : ] = bboxes[i]
    output = (result, ) if not bboxes else (result, bbox_result)
    
    return output

@dataclass
class DataCollatorForMaskedLM:
    """
    Data collator used for masked language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
        pad_token_bbox_value (:obj:`int`, `optional`, defaults to 0):
            The value for padding token bounding box coordinates.
        assign_same_bbox (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to assign the same bbox to each token.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    pad_token_bbox_value: int = 0
    assign_same_bbox: bool = False

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, 
        examples: List[Union[
            List[int], 
            torch.Tensor, 
            Dict[str, torch.Tensor],
            Dict[str, List[int]]
        ]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples, 
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of
            )

            has_bbox_input = "bbox" in batch.keys() and batch["bbox"] is not None
            if has_bbox_input:
                sequence_length = torch.tensor(batch["input_ids"]).shape[1]
                padding_side = self.tokenizer.padding_side
                if padding_side == "right":
                    batch["bbox"] = [bbox + [[self.pad_token_bbox_value] * 4] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
                else:
                    batch["bbox"] = [[[self.pad_token_bbox_value] * 4] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]

            batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        else:
            collate_output = _collate_batch(
                examples, 
                self.tokenizer, 
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_token_bbox_value=self.pad_token_bbox_value
            )
            batch = {"input_ids": collate_output[0]}
            if len(collate_output) == 2:
                batch["bbox"] = collate_output[1]

        if self.assign_same_bbox:
            batch["bbox"] = None 
        
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        pad_token_bbox_value (:obj:`int`, `optional`, defaults to 0):
            The value for padding token bounding box coordinates.
        use_2d_attn_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If attention is restricted with SkimmingMask, prepare a 2-d attention matrix.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    pad_token_bbox_value: int = 0
    use_2d_attn_mask: bool = False

    def __call__(
        self,
        features: List[Union[Dict[str, torch.Tensor], Dict[str, List[int]]]]
    ):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        has_bbox_input = "bbox" in features[0]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if self.use_2d_attn_mask:
            attention_mask_2d = []
            for token_ids in batch["input_ids"]:
                num_padding = token_ids.count(self.tokenizer.pad_token_id)
                attention_mask_2d.append(
                    torch.ones((len(token_ids) - num_padding, len(token_ids) - num_padding))
                )
            

        batch_size, sequence_length = torch.tensor(batch["input_ids"]).shape
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            if labels:
                batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[self.pad_token_bbox_value] * 4] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if self.use_2d_attn_mask:
                batch_2d_mask = attention_mask_2d[0].new_full([batch_size, self.max_length, self.max_length], 0)
                for i in range(batch_size):
                    batch_2d_mask[i, : attention_mask_2d[i].shape[0], : attention_mask_2d[i].shape[0]] = attention_mask_2d[i]
        else:
            if labels:
                batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[self.pad_token_bbox_value] * 4] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if self.use_2d_attn_mask:
                batch_2d_mask = attention_mask_2d[0].new_full([batch_size, self.max_length, self.max_length], 0)
                for i in range(batch_size):
                    batch_2d_mask[i, (self.max_length - attention_mask_2d[i].shape[0]) : , (self.max_length - attention_mask_2d[i].shape[0]) : ] = attention_mask_2d[i]


        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        
        if self.use_2d_attn_mask:
            batch["attention_mask"] = batch_2d_mask

        return batch