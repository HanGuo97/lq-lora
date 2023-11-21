import os
import click
import torch

import run_clm
from peft.tuners import lora
from peft import PeftModelForCausalLM
from models.lora_utils import (
    replace_weight_,
    maybe_sparsify_or_quantize)
from models.quantization_utils import QuantConfig
from experiments.legacy_evaluation_utils import legacy_evaluation


CHECKPOINT_BASE_DIR_DICT = {

}


def transform_lora_adapters(model: PeftModelForCausalLM) -> None:
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError

    qconfig_nf8 = QuantConfig(
        num_bits=8,
        num_bits_0=8,
        num_bits_1="fp32",
        block_size_0=64,
        block_size_1=256)

    click.secho(f"Transforming LoRA adapters.", fg="blue")
    for name, submodule in model.named_modules():
        # This implicitly assumes that `LoraLayer`
        # do not include `LoraLayer` within the module.
        if isinstance(submodule, lora.LoraLayer):
            print(f"{name:<50}")
            if type(submodule) is lora.Linear:
                submodule_lora_A = submodule.lora_A[submodule.active_adapter]
                submodule_lora_B = submodule.lora_B[submodule.active_adapter]
                submodule_lora_A.weight.requires_grad_(False)
                submodule_lora_B.weight.requires_grad_(False)
                qLA = maybe_sparsify_or_quantize(
                    submodule_lora_A.weight,
                    qconfig=qconfig_nf8)
                qLB = maybe_sparsify_or_quantize(
                    submodule_lora_B.weight,
                    qconfig=qconfig_nf8)
                replace_weight_(
                    module=submodule_lora_A,
                    new_weight=qLA)
                replace_weight_(
                    module=submodule_lora_B,
                    new_weight=qLB)
            else:
                raise TypeError


if __name__ == "__main__":

    # Setting up the model, tokenizer
    trainer = run_clm.main(return_trainer=True)

    # Load the model checkpoint
    checkpoint_base_dir = os.getenv(
        "CHECKPOINT_BASE_DIR",
        default=None)
    if checkpoint_base_dir is None:
        checkpoint_base_dir = (
            CHECKPOINT_BASE_DIR_DICT[
                trainer.args.output_dir])

    checkpoint_path = os.path.join(
        checkpoint_base_dir,
        trainer.args.output_dir,
        "full_model.pth")
    state_dict = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"))
    trainer.model.load_state_dict(state_dict)
    click.secho(f"Loaded model from {checkpoint_path}", fg="green")

    # Optionally transforming the adapters
    if os.getenv("TRANSFORM_ADAPTERS", default=False) is not False:
        transform_lora_adapters(trainer.model)

    # Run the evaluation
    results = legacy_evaluation(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        device="cuda")
    for dataset_name, result in results.items():
        click.secho(f"{dataset_name}: {result}", fg="green")
