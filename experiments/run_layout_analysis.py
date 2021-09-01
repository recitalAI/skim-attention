
#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skim import AutoModelForMaskedLM
from sklearn.metrics import classification_report

import torch
from torch import nn

from datasets import ClassLabel, load_dataset

import skim.data.datasets.docbank
import transformers 
from skim.data import DataCollatorForTokenClassification
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import (
    is_main_process,
    get_last_checkpoint,
)
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    core_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "If training <core_model>+SkimEmbeddings or SkimmingMask, "
            "path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    skim_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "If training <core_model>+SkimEmbeddings or SkimmingMask, "
            "path to pretrained Skimformer model."
        }
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    core_model_type: Optional[str] = field(
        default="bert",
        metadata={"help": "Core model type in <core_model>+SkimEmbeddings or SkimmingMask"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    contextualize_2d_positions: bool = field(
        default=False, 
        metadata={"help": "Contextualize the layout embeddings prior to Skim-Attention."}
    )
    top_k: int = field(
        default=0, 
        metadata={"help": "If > 0, SkimmingMask keeps the k-most attended tokens for each token."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    proxy: Optional[str] = field(
        default=None,
        metadata={"help": "Proxy server to use."},
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the data directory."}
    )
    cached_data_dir: str = field(
        default=None,
        metadata={"help": "Path to the cached features"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to 512."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_dataset(
        os.path.abspath(skim.data.datasets.docbank.__file__),
        data_dir=data_args.data_dir,
        cache_dir=data_args.cached_data_dir,
    )
   
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "words" if "words" in column_names else column_names[0]
    label_column_name = "tags" if "tags" in column_names else column_names[1]

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_args.model_type = model_args.model_type.lower()
    model_args.core_model_type = model_args.core_model_type.lower()

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    # initialize config if not BertWithSkimEmbed or SkimmingMask
    if (
        model_args.model_type not in ["bertwithskimembed", "skimmingmask"]
        or (
            model_args.model_type in ["bertwithskimembed", "skimmingmask"]
            and model_args.model_name_or_path
        )
    ):
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            **config_kwargs,
        )
    
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_type in ["longformer", "longskimformer"]:
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        **tokenizer_kwargs,
    )

    if model_args.model_type not in ["bertwithskimembed", "skimmingmask"]:
        logger.info(
            f"Fine-tuning {model_args.model_type} with weights initialized from "
            f"{model_args.model_name_or_path}."
        )
        model_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            # config=config,
            **model_kwargs,
        )
        
        if model.config.num_labels != config.num_labels or model.config.id2label != config.id2label:
            model_architecture = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING[CONFIG_MAPPING[model_args.model_type]]
            if model_architecture == model.config.architectures[0]:
                logger.info(
                    f"`model.config.num_labels` ({model.config.num_labels}) != `config.num_labels` ({config.num_labels}) "
                    "Reseting `model.classifier`"
                )
            model.classifier = nn.Linear(model.config.hidden_size, config.num_labels)
            model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            model.config.num_labels = config.num_labels
            model.num_labels = config.num_labels

    elif model_args.model_type == "bertwithskimembed":
        assert model_args.skim_model_name_or_path and model_args.core_model_name_or_path, "Must provided model checkpoints for "\
            "`skim_model_name_or_path` and `core_model_name_or_path`"
        logger.info(
            f"Fine-tuning BertWithSkimEmbed with 2D position embeddings initialized "\
                f"with weights from {model_args.skim_model_name_or_path} and " \
                    f"core model initialized with weights from {model_args.core_model_name_or_path}"
        )

        skim_model = AutoModelForMaskedLM.from_pretrained(
            model_args.skim_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.skim_model_name_or_path),
            cache_dir=model_args.cache_dir,
        )
        core_model = AutoModelForMaskedLM.from_pretrained(
            model_args.core_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.core_model_name_or_path),
            cache_dir=model_args.cache_dir,
        )

        config = CONFIG_MAPPING[model_args.model_type](
            vocab_size=core_model.config.vocab_size,
            hidden_size=core_model.config.hidden_size,
            hidden_layout_size=skim_model.config.hidden_layout_size,
            num_hidden_layers=core_model.config.num_hidden_layers,
            num_attention_heads=core_model.config.num_attention_heads,
            intermediate_size=core_model.config.intermediate_size,
            hidden_act=core_model.config.hidden_act,
            hidden_dropout_prob=core_model.config.hidden_dropout_prob,
            attention_probs_dropout_prob=core_model.config.attention_probs_dropout_prob,
            max_position_embeddings=core_model.config.max_position_embeddings,
            type_vocab_size=core_model.config.type_vocab_size,
            initializer_range=core_model.config.initializer_range,
            layer_norm_eps=core_model.config.layer_norm_eps,
            pad_token_id=core_model.config.pad_token_id,
            gradient_checkpointing=core_model.config.gradient_checkpointing,
            max_2d_position_embeddings=skim_model.config.max_2d_position_embeddings,
            contextualize_2d_positions=model_args.contextualize_2d_positions,
            num_hidden_layers_layout_encoder=skim_model.config.num_hidden_layers_layout_encoder,
            num_attention_heads_layout_encoder=skim_model.config.num_attention_heads_layout_encoder,
            num_labels=num_labels,
        )
        model = AutoModelForTokenClassification.from_config(config=config)

        # Copy weights
        with torch.no_grad():
            # Copy layout embeddings from Skimformer
            model.bert_with_skim_embed.embeddings.x_position_embeddings.load_state_dict(
                skim_model.skimformer.two_dim_pos_embeddings.x_position_embeddings.state_dict()
            )
            model.bert_with_skim_embed.embeddings.y_position_embeddings.load_state_dict(
                skim_model.skimformer.two_dim_pos_embeddings.y_position_embeddings.state_dict()
            )
            model.bert_with_skim_embed.embeddings.h_position_embeddings.load_state_dict(
                skim_model.skimformer.two_dim_pos_embeddings.h_position_embeddings.state_dict()
            )
            model.bert_with_skim_embed.embeddings.w_position_embeddings.load_state_dict(
                skim_model.skimformer.two_dim_pos_embeddings.w_position_embeddings.state_dict()
            )
        
            # Copy text embeddings and encoder weights from core model               
            model.bert_with_skim_embed.embeddings.word_embeddings.load_state_dict(
                core_model.bert.embeddings.word_embeddings.state_dict()
            )
            model.bert_with_skim_embed.embeddings.position_embeddings.load_state_dict(
                core_model.bert.embeddings.position_embeddings.state_dict()
            )
            model.bert_with_skim_embed.embeddings.token_type_embeddings.load_state_dict(
                core_model.bert.embeddings.token_type_embeddings.state_dict()
            )
            model.bert_with_skim_embed.embeddings.LayerNorm.load_state_dict(
                core_model.bert.embeddings.LayerNorm.state_dict()
            )

            model.bert_with_skim_embed.encoder.load_state_dict(core_model.bert.encoder.state_dict())

    else:
        assert (
            model_args.skim_model_name_or_path and model_args.core_model_name_or_path
        ), f"Must provide `skim_model_name_or_path` and `core_model_name_or_path` to instantiate {model_args.model_type} model."

        logger.info(
            f"Fine-tuning SkimmingMask with layout embeddings and Skim-Attention initialized "\
                f"with weights from {model_args.skim_model_name_or_path}, " \
                    f"core model initialized with weights from {model_args.core_model_name_or_path}" \
                        f" and `top_k` = {model_args.top_k}"
        )

        skim_model = AutoModelForMaskedLM.from_pretrained(
            model_args.skim_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.skim_model_name_or_path),
            cache_dir=model_args.cache_dir,
        )
        core_model = AutoModelForMaskedLM.from_pretrained(
            model_args.core_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.core_model_name_or_path),
            cache_dir=model_args.cache_dir,
        )
        assert model_args.top_k > 0, "`top_k` must be > 0"

        config = CONFIG_MAPPING[model_args.model_type](
            vocab_size=core_model.config.vocab_size,
            hidden_size=core_model.config.hidden_size,
            hidden_layout_size=skim_model.config.hidden_layout_size,
            num_hidden_layers=core_model.config.num_hidden_layers,
            num_attention_heads=core_model.config.num_attention_heads,
            num_layout_attention_heads=skim_model.config.num_attention_heads,
            intermediate_size=core_model.config.intermediate_size,
            hidden_act=core_model.config.hidden_act,
            hidden_dropout_prob=core_model.config.hidden_dropout_prob,
            attention_probs_dropout_prob=core_model.config.attention_probs_dropout_prob,
            max_position_embeddings=core_model.config.max_position_embeddings,
            type_vocab_size=core_model.config.type_vocab_size,
            initializer_range=core_model.config.initializer_range,
            layer_norm_eps=core_model.config.layer_norm_eps,
            skim_attention_head_size=skim_model.config.skim_attention_head_size,
            pad_token_id=core_model.config.pad_token_id,
            gradient_checkpointing=core_model.config.gradient_checkpointing,
            max_2d_position_embeddings=skim_model.config.max_2d_position_embeddings,
            contextualize_2d_positions=skim_model.config.contextualize_2d_positions,
            num_hidden_layers_layout_encoder=skim_model.config.num_hidden_layers_layout_encoder,
            num_attention_heads_layout_encoder=skim_model.config.num_attention_heads_layout_encoder,
            num_labels=num_labels,
            top_k=model_args.top_k,
            core_model_type=model_args.core_model_type,
        )

        model = AutoModelForTokenClassification.from_config(config)

        with torch.no_grad():
            # copy layout embeddings from Skimformer
            model.skimming_mask_model.two_dim_pos_embeddings.load_state_dict(
                skim_model.skimformer.two_dim_pos_embeddings.state_dict()
            )
            
            # copy contextualizer
            if skim_model.config.contextualize_2d_positions:
                logger.info("Contextualizing 2d positions")
                model.skimming_mask_model.layout_encoder.load_state_dict(
                    skim_model.skimformer.layout_encoder.state_dict()
                )
            # copy Skim-Attention
            model.skimming_mask_model.skim_attention.query.load_state_dict(
                skim_model.skimformer.skim_attention.query.state_dict()
            )
            model.skimming_mask_model.skim_attention.key.load_state_dict(
                skim_model.skimformer.skim_attention.key.state_dict()
            )

            # Copy text embeddings and encoder weights from core model
            if model_args.core_model_type == "bert":
                core_model_base = core_model.bert
            else:
                assert model_args.core_model_type == "layoutlm"
                core_model_base = core_model.layoutlm 

            model.skimming_mask_model.embeddings.load_state_dict(
                core_model_base.embeddings.state_dict()
            )
            model.skimming_mask_model.encoder.load_state_dict(core_model_base.encoder.state_dict())
    

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]

            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)
            bboxes.append(bbox_inputs)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        padding=padding,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        use_2d_attn_mask=(model_args.model_type == "skimmingmask"),
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Map from label_id to label
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # flatten lists
        flat_true_predictions = [pred for sublist in true_predictions for pred in sublist]
        flat_true_labels = [label for sublist in true_labels for label in sublist]
        
        labels_to_detect = np.unique(flat_true_labels)

        report = classification_report(
            flat_true_labels, 
            flat_true_predictions, 
            labels=labels_to_detect, 
            output_dict=True,
            zero_division=0,
        )

        results = {}

        for key_label, value_label in sorted(report.items()):
            if type(value_label) != dict:           
                results[key_label] = value_label
            else:
                for key_metric, value_metric in value_label.items():
                    results[key_label + "_" + key_metric] = value_metric
            
        return results

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
