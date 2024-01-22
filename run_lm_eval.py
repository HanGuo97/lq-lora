import argparse
from experiments.lm_eval_utils import run_evaluations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--base_output_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluations(
        model_name_or_path=args.model_name_or_path,
        base_output_path=args.base_output_path,
        checkpoint_dir=args.checkpoint_dir,
    )
