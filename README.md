# Skim-Attention

[Skim-Attention: Learning to Focus via Document Layout](https://arxiv.org/abs/2109.01078) Laura Nguyen, Thomas Scialom, Jacopo Staiano, Benjamin Piwowarski, EMNLP 2021

## Introduction

*Skim-Attention* is a new attention mechanism that takes advantage of the document layout by attending exclusively to the 2-dimensional position of the words. We propose two approaches to exploit *Skim-Attention*: i) *Skimformer*, wherein self-attention is replaced with Skim-Attention; and ii) *SkimmingMask*, where an attention mask is built from Skim-Attention and fed to a Transformer language model.

## Environment Setup

Setup environment as follows:

~~~shell
$ conda create -n skim-attention python=3.8
$ conda activate skim-attention
$ git clone ssh://git@source.recital.ai:2222/research/doc-intelligence/skim-attention.git
$ cd skim-attention
$ pip install -r requirements.txt
$ pip install -e .
~~~

## Pre-train Skimformer 

Pre-train Skimformer as follows:

~~~
$ srun python experiments/run_pretraining.py --output_dir path/to/output/dir \
                                             --max_seq_length 512 \
                                             --data_dir path/to/data/dir \
                                             --model_type skimformer \
                                             --hidden_layout_size 768 \
                                             --skim_attention_head_size 64 \
                                             --tokenizer_name bert-base-uncased \
                                             --use_fast_tokenizer \
                                             --contextualize_2d_positions \
                                             --do_train \
                                             --do_eval \
                                             --do_predict \
                                             --learning_rate 1e-4 \
                                             --max_steps 10000 \
                                             --weight_decay 0.01 \
                                             --warmup_steps 100 \
                                             --per_device_train_batch_size 8 \
                                             --fp16 \
                                             --gradient_accumulation_steps 2 
~~~

To pre-train LongSkimformer, replace `skimformer` with `longskimformer`, and provide a value for the `attention_window` parameter.

## Document Layout Analysis

Skim-Attention is evaluated on a downstream task, document layout analysis. We use a subset of the [DocBank](https://doc-analysis.github.io/docbank-page/index.html) dataset.

To fine-tune Skimformer, run the following:

~~~shell
$ run python experiments/run_layout_analysis.py --output_dir path/to/output/dir \
                                                --max_seq_length 512 \
                                                --data_dir path/to/data/dir \
                                                --model_type skimformer \
                                                --model_name_or_path path/to/pretrained/model \
                                                --tokenizer_name bert-base-uncased \
                                                --use_fast_tokenizer \
                                                --do_train \
                                                --do_eval \
                                                --do_predict \
                                                --num_train_epochs 10 \
                                                --learning_rate 5e-5 \
                                                --per_device_train_batch_size 8 \
                                                --fp16 \
                                                --gradient_accumulation_steps 2 
~~~

To plug Skimformer's layout embeddings in BERT, do as follows:

~~~shell
$ run python experiments/run_layout_analysis.py --output_dir path/to/output/dir \
                                                --max_seq_length 512 \
                                                --data_dir path/to/data/dir \
                                                --model_type bertwithskimembed \
                                                --skim_model_name_or_path path/to/pretrained/skimformer/model \
                                                --core_model_type bert \
                                                --core_model_name_or_path path/to/pretrained/bert/model \
                                                --tokenizer_name bert-base-uncased \
                                                --use_fast_tokenizer \
                                                --do_train \
                                                --do_eval \
                                                --do_predict \
                                                --num_train_epochs 10 \
                                                --learning_rate 5e-5 \
                                                --per_device_train_batch_size 8 \
                                                --fp16 \
                                                --gradient_accumulation_steps 2 
~~~

To plug SkimmingMask in BERT or LayoutLM, do as follows:

~~~shell
$ run python experiments/run_layout_analysis.py --output_dir path/to/output/dir \
                                                --max_seq_length 512 \
                                                --data_dir path/to/data/dir \
                                                --model_type skimmingmask \
                                                --skim_model_name_or_path path/to/pretrained/skimformer/model \
                                                --core_model_type <bert | layoutlm> \
                                                --core_model_name_or_path path/to/pretrained/bert/or/layoutlm/model \
                                                --tokenizer_name <bert-base-uncased | microsoft/layoutlm-base-uncased> \
                                                --use_fast_tokenizer \
                                                --top_k 128 \
                                                --do_train \
                                                --do_eval \
                                                --do_predict \
                                                --num_train_epochs 10 \
                                                --learning_rate 5e-5 \
                                                --per_device_train_batch_size 8 \
                                                --fp16 \
                                                --gradient_accumulation_steps 2 
~~~

## Citation

``` latex
@article{nguyen2021skimattention,
    title={Skim-Attention: Learning to Focus via Document Layout}, 
    author={Laura Nguyen and Thomas Scialom and Jacopo Staiano and Benjamin Piwowarski},
    journal={arXiv preprint arXiv:2109.01078}
    year={2021},
}
```
