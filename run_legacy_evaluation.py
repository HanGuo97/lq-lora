import os
import click
import torch

import run_clm
from models.lora_utils import transform_lora_adapters_nf8
from experiments.legacy_evaluation_utils import legacy_evaluation


CHECKPOINT_BASE_DIR_DICT = {

}


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
        transform_lora_adapters_nf8(trainer.model)

    # Run the evaluation
    results = legacy_evaluation(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        device="cuda")
    for dataset_name, result in results.items():
        click.secho(f"{dataset_name}: {result}", fg="green")
