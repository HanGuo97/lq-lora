import os
import click
import torch
import argparse
import dataclasses
import transformers
from typing import Tuple
from peft import PeftModelForCausalLM
from transformers import PreTrainedTokenizerBase
from experiments.vicuna_utils import run_eval
from run_oasst1 import (
    get_accelerate_model,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
    CustomAdditionalArguments)


def setup() -> Tuple[PeftModelForCausalLM, PreTrainedTokenizerBase, argparse.Namespace]:
    hfparser = transformers.HfArgumentParser((  # type: ignore
        ModelArguments,
        DataArguments,
        TrainingArguments,
        GenerationArguments,
        CustomAdditionalArguments))

    (model_args,
     data_args,
     training_args,
     generation_args,
     custom_args,
     _) = hfparser.parse_args_into_dataclasses(
         return_remaining_strings=True)

    # training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    training_args = dataclasses.replace(
        training_args,
        generation_config=transformers.GenerationConfig(
            **vars(generation_args)))
    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
        **vars(custom_args))
    print(args)

    model, tokenizer = get_accelerate_model(args)

    model.config.use_cache = False
    return model, tokenizer, args


if __name__ == "__main__":

    # Setting up the model, tokenizer
    model, tokenizer, args = setup()
    model.to(device="cuda")

    # Load the model checkpoint
    checkpoint_base_dir = os.getenv(
        "CHECKPOINT_BASE_DIR",
        default=None)
    if checkpoint_base_dir is None:
        raise ValueError
    checkpoint_path = os.path.join(
        checkpoint_base_dir,
        args.output_dir,
        "full_model.pth")
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    click.secho(f"Loaded model from {checkpoint_path}", fg="green")

    # Run the evaluation
    run_eval(
        model=model,
        tokenizer=tokenizer,
        model_id="guanaco",
        question_file="data/vicuna_question.jsonl",
        answer_file=f"{args.output_dir}_answer.jsonl",
        num_gpus=1,
    )
