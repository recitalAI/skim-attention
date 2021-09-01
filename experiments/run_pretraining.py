#!/usr/bin/env python
# coding=utf-8

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

import skim.data.datasets.mix
import transformers
from skim.data import DataCollatorForMaskedLM
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
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

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. "
            "Leave None if you want to train LayoutLM from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES) + ", layoutlm"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_type"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_type"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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

    use_1d_positions: bool = field(
        default=False, metadata={"help": "Use sequential positions for position-based attention"}
    )
    degrade_2d_positions: bool = field(
        default=False, metadata={"help": "Replace the bbox coordinates with the bbox center"}
    )
    contextualize_2d_positions: bool = field(
        default=False, metadata={"help": "Pass the bbox embeddings through a small BERT encoder before feeding them to the layout layer."}
    )
    hidden_layout_size: int = field(
        default=768, metadata={"help": "Hidden size for 2D positional embeddings."}
    )
    skim_attention_head_size: int = field(
        default=64, metadata={"help": "Dimensionality of the projected key, query and value vectors in Skim-Attention."}
    )
    attention_head_size: int = field(
        default=64, metadata={"help": "Dimensionality of the projected key, query and value vectors in standard attention."}
    )
    attention_window: int = field(
        default=256, metadata={"help": "Size of the attention window around each token."}
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to the data directory."}
    )
    cached_data_dir: str = field(
        default=None,
        metadata={"help": "Path to the cached datasets"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached sets"}
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_token_bbox_value: int = field(
        default=0, metadata={"help": "Value for PAD token bounding box"}
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

    assign_same_bbox: bool = field(
        default=False, metadata={"help": "Assign same bbox to each token"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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

    if not os.path.exists(data_args.data_dir) or not os.listdir(data_args.data_dir):
        raise ValueError("Supply a valid path to --data_dir.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_dataset(
        os.path.abspath(skim.data.datasets.mix.__file__),
        data_dir=data_args.data_dir,
        cache_dir=data_args.cached_data_dir,
    )

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["test"].column_names
    text_column_name = "words" if "words" in column_names else column_names[0]
    
    remove_columns = column_names

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    model_args.model_type = model_args.model_type.lower()

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config_class = CONFIG_MAPPING[model_args.model_type]
        if model_args.model_type == "skimformer":
            config = config_class(
                hidden_layout_size=model_args.hidden_layout_size, 
                skim_attention_head_size=model_args.skim_attention_head_size,
                attention_head_size=model_args.attention_head_size,
                use_1d_positions=model_args.use_1d_positions,
                degrade_2d_positions=model_args.degrade_2d_positions,
                contextualize_2d_positions=model_args.contextualize_2d_positions,
            )
        elif model_args.model_type == "longskimformer":
            config = config_class(
                hidden_layout_size=model_args.hidden_layout_size, 
                skim_attention_head_size=model_args.skim_attention_head_size,
                attention_head_size=model_args.attention_head_size,
                use_1d_positions=model_args.use_1d_positions,
                degrade_2d_positions=model_args.degrade_2d_positions,
                contextualize_2d_positions=model_args.contextualize_2d_positions,
                attention_window=model_args.attention_window,
            )
        else:
            config = config_class()
            logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_type in ["longformer", "longskimformer"]:
        tokenizer_kwargs["add_prefix_space"] = True
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info(f"Training new {model_args.model_type} model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the bboxes with them.
    def tokenize_and_align_bboxes(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        bboxes = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            bbox = examples["bboxes"][org_batch_index]

            previous_word_idx = None
            bbox_inputs = []
            for word_idx in word_ids:
                if word_idx is None:
                    bbox_inputs.append([0, 0, 0, 0])
                elif word_idx != previous_word_idx:
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            bboxes.append(bbox_inputs)
        tokenized_inputs["bbox"] = bboxes
        return tokenized_inputs

    # Get datasets
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_bboxes,
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
            tokenize_and_align_bboxes,
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
            tokenize_and_align_bboxes,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    data_collator = DataCollatorForMaskedLM(
        tokenizer,
        mlm_probability=data_args.mlm_probability,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        pad_token_bbox_value=0,
        assign_same_bbox=data_args.assign_same_bbox,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model() # Saves the tokenizer too for easy upload

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
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        output_eval_ppl_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_ppl_file, "w") as writer:
                logger.info("***** Eval results *****")
                logger.info("  ppl = %s", str(metrics["perplexity"]))
                writer.write("ppl = %s\n" % str(metrics["perplexity"]))

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        metrics = trainer.evaluate(test_dataset)

        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        output_test_ppl_file = os.path.join(training_args.output_dir, "test_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_test_ppl_file, "w") as writer:
                logger.info("***** Test results *****")
                logger.info("  ppl = %s", str(metrics["perplexity"]))
                writer.write("ppl = %s\n" % str(metrics["perplexity"]))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
