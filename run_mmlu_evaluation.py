import os
import click
import torch

import run_clm
from experiments.mmlu_utils import run_evaluation


CHECKPOINT_BASE_DIR_DICT = {
    "output_c4_lora_20230909_ranks64_7b_nf3"        : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lora_20230909_ranks64_7b_nf4"        : "/export/share/experiments/20230912/795acd350b74/",

    "output_c4_lora_20230909_ranks64_7b_gptq-3bit"  : "/export/share/experiments/20230916/5290946275a0/",
    "output_c4_lora_20230909_ranks64_7b_gptq-4bit"  : "/export/share/experiments/20230916/5290946275a0/",

    "output_c4_lora_20230909_ranks64_7b_None_2.5"   : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lora_20230909_ranks64_7b_None_2.75"  : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lora_20230909_ranks64_7b_None_3"     : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lora_20230909_ranks64_7b_None_3.25"  : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lora_20230909_ranks64_7b_None_3.5"   : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lora_20230909_ranks64_7b_None_3.75"  : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lora_20230909_ranks64_7b_None_4"     : "/export/share/experiments/20230912/ab23a4f35ea4/",

    "output_c4_lpq_20230909_ranks64_7b_None_2.5"    : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_None_2.75"   : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_None_3"      : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_None_3.25"   : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_None_3.5"    : "/export/share/experiments/20230914/795acd350b74/",
    "output_c4_lpq_20230909_ranks64_7b_None_3.75"   : "/export/share/experiments/20230912/795acd350b74/",
    "output_c4_lpq_20230909_ranks64_7b_None_4"      : "/export/share/experiments/20230912/795acd350b74/",

    "output_c4_lpq_20230909_ranks64_7b_c4_2.5"      : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_c4_2.75"     : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_c4_3"        : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_c4_3.25"     : "/export/share/experiments/20230912/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_7b_c4_3.5"      : "/export/share/experiments/20230914/795acd350b74/",
    "output_c4_lpq_20230909_ranks64_7b_c4_3.75"     : "/export/share/experiments/20230912/795acd350b74/",
    "output_c4_lpq_20230909_ranks64_7b_c4_4"        : "/export/share/experiments/20230912/795acd350b74/",

    # 70B
    "output_c4_lora_20230909_ranks64_70b_nf3"       : "/export/share/experiments/20230922/b7a0cbd20ed7/",
    "output_c4_lora_20230909_ranks64_70b_nf4"       : "/export/share3/experiments/20230927/ab23a4f35ea4/",

    "output_c4_lora_20230909_ranks64_70b_None_2.5"  : "/export/share3/experiments/20230927/24053ff618b3/",
    "output_c4_lora_20230909_ranks64_70b_None_2.75" : "/export/share3/experiments/20230927/24053ff618b3/",
    "output_c4_lora_20230909_ranks64_70b_None_3"    : "/export/share3/experiments/20230925/bc9575f2d323/",
    "output_c4_lora_20230909_ranks64_70b_None_3.25" : "/export/share3/experiments/20230922/21f9d4382283/",
    "output_c4_lora_20230909_ranks64_70b_None_3.5"  : "/export/share3/experiments/20230922/21f9d4382283/",
    "output_c4_lora_20230909_ranks64_70b_None_3.75" : "/export/share3/experiments/20230922/21f9d4382283/",
    "output_c4_lora_20230909_ranks64_70b_None_4"    : "/export/share3/experiments/20230922/21f9d4382283/",

    "output_c4_lpq_20230909_ranks64_70b_None_2.5"   : "/export/share3/experiments/20230925/bc9575f2d323/",
    "output_c4_lpq_20230909_ranks64_70b_None_2.75"  : "/export/share3/experiments/20230925/bc9575f2d323/",
    "output_c4_lpq_20230909_ranks64_70b_None_3"     : "/export/share3/experiments/20230925/bc9575f2d323/",
    "output_c4_lpq_20230909_ranks64_70b_None_3.25"  : "/export/share3/experiments/20230925/bc9575f2d323/",
    "output_c4_lpq_20230909_ranks64_70b_None_3.5"   : "/export/share3/experiments/20230925/bc9575f2d323/",
    "output_c4_lpq_20230909_ranks64_70b_None_3.75"  : "/export/share3/experiments/20230925/bc9575f2d323/",
    "output_c4_lpq_20230909_ranks64_70b_None_4"     : "/export/share3/experiments/20230925/bc9575f2d323/",

    "output_c4_lpq_20230909_ranks64_70b_c4_2.5"     : "/export/share/experiments/20230922/b7a0cbd20ed7/",
    "output_c4_lpq_20230909_ranks64_70b_c4_2.75"    : "/export/share/experiments/20230922/b7a0cbd20ed7/",
    "output_c4_lpq_20230909_ranks64_70b_c4_3"       : "/export/share/experiments/20230922/b7a0cbd20ed7/",
    "output_c4_lpq_20230909_ranks64_70b_c4_3.25"    : "/export/share3/experiments/20230927/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_70b_c4_3.5"     : "/export/share3/experiments/20230927/ab23a4f35ea4/",
    "output_c4_lpq_20230909_ranks64_70b_c4_3.75"    : "/export/share/experiments/20230922/b7a0cbd20ed7/",
    "output_c4_lpq_20230909_ranks64_70b_c4_4"       : "/export/share3/experiments/20230927/ab23a4f35ea4/",
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

    # Run the evaluation
    run_evaluation(trainer=trainer)
