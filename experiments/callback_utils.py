import os
import click
import torch
from typing import Union
from transformers import (
    TrainerState,
    TrainerControl,
    TrainerCallback,
    TrainingArguments)
from peft import (
    PeftModelForCausalLM,
    PeftModelForSequenceClassification)


class SaveFullModelCallback(TrainerCallback):

    def _save_full_model(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model: Union[PeftModelForCausalLM, PeftModelForSequenceClassification],
    ) -> None:
        if not isinstance(model, (PeftModelForCausalLM, PeftModelForSequenceClassification)):
            raise TypeError(
                f"Expected `PeftModelForCausalLM`, or "
                f"`PeftModelForSequenceClassification`, "
                f"but got {type(model)}")
        if not args.should_save:
            return

        state_dict = model.state_dict()
        file_name = os.path.join(
            args.output_dir,
            f"full_model-{state.global_step}.pth")
        torch.save(state_dict, file_name)
        click.secho(f"Saved model state dict to {file_name}", fg="green")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._save_full_model(
            args=args,
            state=state,
            model=kwargs["model"])
