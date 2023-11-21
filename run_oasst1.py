# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import click
import torch
import argparse
import dataclasses
import numpy as np
import transformers
from tqdm import tqdm
from typing import Tuple, Optional

import evaluate
from datasets import load_dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    LlamaTokenizer)
from peft import PeftModelForCausalLM
from peft.tuners.lora import LoraLayer

from models import lora_utils
from experiments import vicuna_utils
from experiments import callback_utils
from experiments.qlora import (
    logger,
    IGNORE_INDEX,
    DEFAULT_PAD_TOKEN,
    DataArguments,
    ModelArguments,
    TrainingArguments,
    GenerationArguments,
    SavePeftModelCallback,
    make_data_module,
    print_trainable_parameters,
    smart_tokenizer_and_embedding_resize)


# This number is set so that it's consistent with other experiments.
SEQUENCE_LENGTH = 1024
MMLU_DATA_BASE_DIR = "/export/share/experiments/20230903/qlora/"


@dataclasses.dataclass
class CustomAdditionalArguments(object):
    lora_config: Optional[str] = dataclasses.field(default=None)
    lora_model_name: Optional[str] = dataclasses.field(default=None)
    hf_quantization_method: Optional[str] = dataclasses.field(default=None)


def get_accelerate_model(args: argparse.Namespace) -> Tuple[PeftModelForCausalLM, LlamaTokenizer]:

    if args.full_finetune:
        raise NotImplementedError

    print(f"loading base model {args.model_name_or_path}...")
    hf_quantization_kwargs = lora_utils.get_hf_quantization_config(
        method=args.hf_quantization_method,
        sequence_length=SEQUENCE_LENGTH)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
        torch_dtype=None,
        low_cpu_mem_usage=True,
        **hf_quantization_kwargs)
    # https://github.com/huggingface/transformers/pull/24906
    if model.config.pretraining_tp != 1:
        raise NotImplementedError

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        # ----- Probably doesn't matter -----
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type="llama" if "llama" in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print("Adding special tokens.")
        tokenizer.add_special_tokens({  # type: ignore
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id  # type: ignore
                if model.config.pad_token_id not in [-1, None]
                else tokenizer.pad_token_id
            ),
        })

    if args.lora_config in ["lora", "lora-lpq"]:
        click.secho(f"LoRA Finetuning with `{args.lora_config}`", bg="yellow")
        if not all([
            args.lora_config is not None,
            args.lora_model_name is not None,
            args.hf_quantization_method is None]):
            raise ValueError

        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_gradient_checkpointing=args.gradient_checkpointing)
        lora_utils.transform_lora_layers(
            lpq=(args.lora_config == "lora-lpq"),
            model=model,
            model_name=args.lora_model_name,
            device="cuda")

    elif args.lora_config in ["lora-gptq"]:
        click.secho(f"GPTQ-LoRA Finetuning with `{args.lora_config}`", bg="yellow")
        if not all([
            args.lora_config is not None,
            args.lora_model_name is None,
            args.hf_quantization_method is not None]):
            raise ValueError

        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_gradient_checkpointing=args.gradient_checkpointing)

    elif os.path.isdir(args.lora_config):
        click.secho(f"Continued LoRA Finetuning from `{args.lora_config}`", bg="yellow")
        if not all([
            args.lora_config is not None,
            args.lora_model_name is None,
            args.hf_quantization_method is None]):
            raise ValueError

        # No need to apply `transform_lora_layers` because
        # these will be loaded from the checkpoint.
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_gradient_checkpointing=args.gradient_checkpointing,
            checkpoint_dir=args.lora_config,
            checkpoint_preprocess_embedding=True)

    else:
        click.secho(f"Full Finetuning", bg="yellow")

    for name, module in model.named_modules():
        # if isinstance(module, LoraLayer):
        #     if args.bf16:
        #         module = module.to(dtype=torch.bfloat16)
        if "norm" in name:
            # module = module.to(dtype=torch.float32)
            if module.weight.dtype != torch.float32:
                raise TypeError
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                # if args.bf16 and module.weight.dtype == torch.float32:
                #     module = module.to(dtype=torch.bfloat16)
                if module.weight.dtype != torch.float32:
                    raise TypeError

    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError
    if not isinstance(tokenizer, LlamaTokenizer):
        raise TypeError
    return model, tokenizer


def train() -> None:
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
    print("loaded model")
    set_seed(args.seed)

    data_module = make_data_module(
        tokenizer=tokenizer,
        args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != "predict_dataset"},
        callbacks=[callback_utils.SaveFullModelCallback],
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    else:
        raise NotImplementedError

    if args.do_mmlu_eval:
        if args.mmlu_dataset == "mmlu-zs":
            mmlu_dataset = load_dataset("json", data_files={
                "eval": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/zero_shot_mmlu_val.json"),
                "test": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/zero_shot_mmlu_test.json"),
            })
            mmlu_dataset = mmlu_dataset.remove_columns("subject")
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == "mmlu" or args.mmlu_dataset == "mmlu-fs":
            mmlu_dataset = load_dataset("json", data_files={
                "eval": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/five_shot_mmlu_val.json"),
                "test": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/five_shot_mmlu_test.json"),
            })
            # mmlu_dataset = mmlu_dataset.remove_columns("subject")
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch["labels"][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {"mmlu_loss":loss_mmlu/len(data_loader)}
                subject = mmlu_dataset["subject"]
                subjects = {s:{"refs":[], "preds":[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]["preds"].append(p)
                    subjects[s]["refs"].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]["refs"],
                        predictions=subjects[subject]["preds"]
                    )["accuracy"]
                    results[f"mmlu_{args.mmlu_split}_accuracy_{subject}"] = subject_score
                    subject_scores.append(subject_score)
                results[f"mmlu_{args.mmlu_split}_accuracy"] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

        # Save the full model
        lora_utils.save_full_model(trainer)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

        # # Generate outputs for Vicuna evaluation
        # click.secho("Vicuna Evaluation", fg="blue")
        # if not all([
        #     trainer.model is model,
        #     trainer.tokenizer is tokenizer]):
        #     raise ValueError

        # vicuna_utils.run_eval(
        #     model=trainer.model,
        #     tokenizer=trainer.tokenizer,
        #     model_id="guanaco",
        #     question_file="data/vicuna_question.jsonl",
        #     answer_file=f"{args.output_dir}_answer.jsonl",
        #     num_gpus=1,
        # )

    # Prediction
    if args.do_predict:
        raise NotImplementedError

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()
