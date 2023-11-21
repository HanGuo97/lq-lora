import os
import sys
import click
import torch
from typing import Optional, cast

import run_clm
import run_mlm
from models.allocation_utils import (
    compute_empirical_Fisher_LLaMA,
    prepare_data_for_qconfig_assignments_LLaMA,
    prepare_model_inputs_for_qconfig_assignments_LLaMA)
from models.allocation_utils_2 import (
    compute_empirical_Fisher_RoBERTa,
    prepare_data_for_qconfig_assignments_RoBERTa,
    prepare_model_inputs_for_qconfig_assignments_RoBERTa)

MODEL_PATHS_DICT = {
    "llama-2-7b": "/export/share3/experiments/20230731/llama-2/Llama-2-7b-hf",
    "llama-2-13b": "/export/share3/experiments/20230731/llama-2/Llama-2-13b-hf",
    "llama-2-70b": "/export/share3/experiments/20230731/llama-2/Llama-2-70b-hf",
    "llama-2-7b-512": "/export/share3/experiments/20230731/llama-2/Llama-2-7b-hf",
    "llama-2-7b-1024": "/export/share3/experiments/20230731/llama-2/Llama-2-7b-hf",
    "llama-2-70b-1024": "/export/share3/experiments/20230731/llama-2/Llama-2-70b-hf",
}

INPUTS_PATHS_DICT = {
    "llama-2-7b": "/export/share/experiments/20230816/fisher_dict_fp32/llama-2-7b.fisher_dict.c4.length-256.nsamples-10000.pth",
    "llama-2-13b": "/export/share/experiments/20230816/fisher_dict_fp32/llama-2-13b.fisher_dict.c4.length-256.nsamples-10000.pth",
    "llama-2-70b": "/export/share/experiments/20230816/fisher_dict_bf16/llama-2-70b.fisher_dict.c4.length-256.nsamples-10000.pth",
    "llama-2-7b-512": "/export/share/experiments/20230906/fisher_dict_fp32/llama-2-7b.fisher_dict.c4.length-512.nsamples-10000.pth",
    "llama-2-7b-1024": "/export/share/experiments/20230909/fisher_dict_fp32/llama-2-7b.fisher_dict.c4.length-1024.nsamples-10000.pth",
    "llama-2-70b-1024": "/export/share/experiments/20230909/fisher_dict_bf16/llama-2-70b.fisher_dict.c4.length-1024.nsamples-10000.pth",
    "roberta-large": "/export/share/experiments/20230923/fisher_dict_fp32/roberta-large.fisher_dict.c4.length-default.nsamples-10000.pth",
}


def prepare_argv_LLaMA(model_name: str, block_size: int) -> None:
    sys.argv = f"""
    run_clm.py \
        --model_name_or_path {MODEL_PATHS_DICT[model_name]} \
        --dataset_name c4 \
        --block_size {block_size} \
        --bf16 True \
        --output_dir tmp_dir \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --evaluation_strategy steps \
        --eval_steps 500 \
        --save_strategy steps \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 3e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --do_train \
        --do_eval \
        --tf32 True
        """.split()
    print(sys.argv)


def prepare_data_for_fisher_LLaMA(
    save_dir: str,
    model_name: str,
    block_size: int = 1024,
    num_examples: int = 10000,
) -> None:
    prepare_argv_LLaMA(
        model_name=model_name,
        block_size=block_size)
    trainer = run_clm.main(return_trainer=True)
    trainer = cast(run_clm.Trainer, trainer)
    model_inputs = prepare_model_inputs_for_qconfig_assignments_LLaMA(
        trainer,
        num_batches=num_examples)
    fisher_dict = compute_empirical_Fisher_LLaMA(
        model=trainer.model,
        inputs=model_inputs,
        dtype=torch.float32,
        device=None)
    save_path = os.path.join(
        save_dir,
        f"{model_name}.fisher_dict.c4.length-{block_size}.nsamples-{num_examples}.pth")
    torch.save(fisher_dict, save_path)
    click.secho(f"Saved Fisher to {save_path}", fg="green")


def prepare_data_for_ilp_LLaMA(
    save_dir: str,
    model_name: str,
    num_ranks: int = 64,
    num_partitions: int = 4,
) -> None:
    inputs_path = INPUTS_PATHS_DICT[model_name]
    for _inputs_path in [inputs_path, None]:
        for _num_ranks in [num_ranks, None]:
            save_path = os.path.join(
                save_dir,
                f"{model_name}.ilp."
                f"ranks-{_num_ranks}."
                f"data-{_inputs_path is not None}.pth")
            prepare_data_for_qconfig_assignments_LLaMA(
                model_path=MODEL_PATHS_DICT[model_name],
                inputs_path=_inputs_path,
                num_ranks=_num_ranks,
                num_partitions=num_partitions,
                save_path=save_path)


def prepare_argv_RoBERTa(model_name: str) -> None:
    sys.argv = f"""
    run_mlm.py \
        --model_name_or_path {model_name} \
        --dataset_name c4 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --do_train \
        --do_eval \
        --output_dir tmp_dir
        """.split()
    print(sys.argv)


def prepare_data_for_fisher_RoBERTa(
    save_dir: str,
    model_name: Optional[str] = None,
    num_examples: int = 10000,
) -> None:
    if model_name is None:
        model_name = "roberta-large"
    # We will use the default `max_seq_length` here instead.
    prepare_argv_RoBERTa(model_name=model_name)
    trainer = run_mlm.main(return_trainer=True)
    trainer = cast(run_mlm.Trainer, trainer)
    model_inputs = prepare_model_inputs_for_qconfig_assignments_RoBERTa(
        trainer,
        num_batches=num_examples)
    fisher_dict = compute_empirical_Fisher_RoBERTa(
        model=trainer.model,
        inputs=model_inputs,
        dtype=torch.float32,
        device=None)
    save_path = os.path.join(
        save_dir,
        f"{model_name}.fisher_dict.c4.length-default.nsamples-{num_examples}.pth")
    torch.save(fisher_dict, save_path)
    click.secho(f"Saved Fisher to {save_path}", fg="green")


def prepare_data_for_ilp_RoBERTa(
    save_dir: str,
    model_name: Optional[str] = None,
    num_ranks: int = 64,
    num_partitions: int = 4,
) -> None:
    if model_name is None:
        model_name = "roberta-large"
    inputs_path = INPUTS_PATHS_DICT[model_name]
    for _inputs_path in [inputs_path, None]:
        for _num_ranks in [num_ranks, None]:
            save_path = os.path.join(
                save_dir,
                f"{model_name}.ilp."
                f"ranks-{_num_ranks}."
                f"data-{_inputs_path is not None}.pth")
            prepare_data_for_qconfig_assignments_RoBERTa(
                model_path=model_name,
                inputs_path=_inputs_path,
                num_ranks=_num_ranks,
                num_partitions=num_partitions,
                save_path=save_path)
