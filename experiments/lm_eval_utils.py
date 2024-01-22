import os
import json
import torch
import logging
import numpy as np
import lm_eval.utils
import lm_eval.tasks
import lm_eval.models
import lm_eval.evaluator
from typing import Optional, List, Dict
from models import lora_utils

logging.getLogger("openai").setLevel(logging.WARNING)
DEFAULT_BATCH_SIZE = 1
DEFAULT_TRANSFORM_ADAPTERS = True
EVALUATION_CONFIGS = [
    {
        "task": "arc_challenge",
        "metric": "acc_norm",
        "num_fewshot": 25,
    },
    {
        "task": "hellaswag",
        "metric": "acc_norm",
        "num_fewshot": 10,
    },
    {
        "task": "truthfulqa_mc",
        "metric": "mc2",
        "num_fewshot": 0,
    },
    {
        "task": "hendrycksTest-*",
        "metric": "acc",
        "num_fewshot": 5,
    },
    {
        "task": "winogrande",
        "metric": "acc",
        "num_fewshot": 5,
    },
    {
        "task": "gsm8k",
        "metric": "acc",
        "num_fewshot": 5,
    },
]


def create_lm_eval_model(
    model_name_or_path: str,
    batch_size: int,
    checkpoint_dir: Optional[str],
    enable_tf32: bool = True,
) -> lm_eval.models.huggingface.AutoCausalLM:

    if checkpoint_dir is not None:
        # We first dispatch them onto CPU, quantize the weights,
        # add adapters, and then move them onto one GPU
        additional_kwargs = {"device_map_option": "cpu"}
    else:
        # For non-quantized models, we will directly dispatch
        # them onto multiple GPUs
        additional_kwargs = {}

    wrapped_model = (
        lm_eval
        .models
        .get_model("hf-causal-experimental")
        .create_from_arg_string(
            f"pretrained={model_name_or_path},use_accelerate=True",
            {
                "batch_size": batch_size,
                "max_batch_size": None,
                "dtype": torch.float32,  # legacy reasons
                "device": None,
                **additional_kwargs,
            }
        )
    )

    for pname, pdevice in wrapped_model.model.hf_device_map.items():
        print(f"Device map: {pname} -> {pdevice}")

    if checkpoint_dir is not None:
        # load the adapted model
        wrapped_model.model = lora_utils.prepare_model_for_lora(
            model=wrapped_model.model,
            num_ranks=64,  # will be ignored
            lora_dropout=0.0,  # will be ignored
            use_gradient_checkpointing=True,
            checkpoint_dir=checkpoint_dir)

        # quantize the adapters
        if DEFAULT_TRANSFORM_ADAPTERS is True:
            lora_utils.transform_lora_adapters_nf8(
                model=wrapped_model.model)

        # move the model onto GPU
        if wrapped_model._device != "cuda":
            raise ValueError(f"_device: {wrapped_model._device}")
        wrapped_model.model.to(wrapped_model._device)

    if enable_tf32 is True:
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        # Some of the function calls above might import modules
        # that disable TF32, so we need to re-enable it here.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    wrapped_model.model.eval()
    return wrapped_model


def evaluate_one(
    task: str,
    model: lm_eval.models.huggingface.AutoCausalLM,
    batch_size: int,
    num_fewshot: int,
    output_path: str,
) -> None:

    task_names = lm_eval.utils.pattern_match(
        patterns=[task],
        source_list=lm_eval.tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")
    results = lm_eval.evaluator.simple_evaluate(
        model=model,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        no_cache=True)

    dumped = json.dumps(results, indent=2)
    print(dumped)

    # Write out results
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(f"num_fewshot: {num_fewshot}, "
          f"batch_size: {batch_size}"
          f"{f' ({batch_sizes})' if batch_sizes else ''}")
    print(lm_eval.evaluator.make_table(results))


def evaluate_all(
    model: lm_eval.models.huggingface.AutoCausalLM,
    batch_size: int,
    base_output_path: str,
) -> None:
    for config in EVALUATION_CONFIGS:
        output_path = f'{base_output_path}-{config["task"]}'
        evaluate_one(
            task=config["task"],
            model=model,
            batch_size=batch_size,
            num_fewshot=config["num_fewshot"],
            output_path=output_path)


def run_evaluations(
    model_name_or_path: str,
    base_output_path: str,
    checkpoint_dir: Optional[str] = None,
) -> None:
    model = create_lm_eval_model(
        model_name_or_path=model_name_or_path,
        batch_size=DEFAULT_BATCH_SIZE,
        checkpoint_dir=checkpoint_dir)
    evaluate_all(
        model=model,
        batch_size=DEFAULT_BATCH_SIZE,
        base_output_path=base_output_path)


def collect_evaluations(
    base_output_path: str,
) -> List[Dict[str, float]]:
    scores = []
    for config in EVALUATION_CONFIGS:
        output_path = f'{base_output_path}-{config["task"]}'
        with open(output_path) as f:
            results = json.load(f)["results"]

        if len(results) == 1:
            score = results[config["task"]][config["metric"]]
        else:
            score = np.mean([
                result[config["metric"]]
                for result in results.values()]).item()

        scores.append({
            "task": config["task"],
            "score": score * 100.,
        })
    return scores
