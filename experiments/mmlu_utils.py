import os
import click
import torch
import evaluate
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Optional
from datasets import load_dataset
from transformers import (
    Trainer,
    Seq2SeqTrainer,
    PreTrainedTokenizerBase,
    Seq2SeqTrainingArguments)

from run_oasst1 import (
    IGNORE_INDEX,
    MMLU_DATA_BASE_DIR)
from experiments.qlora import (
    DataCollatorForCausalLM)


def _run_evaluation(
    trainer: Seq2SeqTrainer,
    tokenizer: PreTrainedTokenizerBase,
    mmlu_name: Optional[str] = None,
    mmlu_split: Optional[str] = None,
    max_mmlu_samples: Optional[int] = None,
    mmlu_source_max_len: int = 2048,
) -> Dict[str, float]:
    # Defaults
    if mmlu_name is None:
        mmlu_name = "mmlu-fs"
    if mmlu_split is None:
        mmlu_split = "test"

    if mmlu_name == "mmlu-zs":
        mmlu_dataset = load_dataset("json", data_files={
            "eval": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/zero_shot_mmlu_val.json"),
            "test": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/zero_shot_mmlu_test.json"),
        })
        mmlu_dataset = mmlu_dataset.remove_columns("subject")
    # MMLU Five-shot (Eval/Test only)
    elif mmlu_name == "mmlu" or mmlu_name == "mmlu-fs":
        mmlu_dataset = load_dataset("json", data_files={
            "eval": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/five_shot_mmlu_val.json"),
            "test": os.path.join(MMLU_DATA_BASE_DIR, "data/mmlu/five_shot_mmlu_test.json"),
        })
        # mmlu_dataset = mmlu_dataset.remove_columns("subject")
    mmlu_dataset = mmlu_dataset[mmlu_split]
    if max_mmlu_samples is not None:
        mmlu_dataset = mmlu_dataset.select(range(max_mmlu_samples))
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
    ]
    accuracy = evaluate.load("accuracy")

    def on_evaluate() -> Dict[str, float]:
        data_loader = trainer.get_eval_dataloader(mmlu_dataset)
        # source_max_len = trainer.data_collator.source_max_len
        # trainer.data_collator.source_max_len = args.mmlu_source_max_len
        if trainer.data_collator.source_max_len != mmlu_source_max_len:
            raise ValueError

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
            results[f"mmlu_{mmlu_split}_accuracy_{subject}"] = subject_score
            subject_scores.append(subject_score)
        results[f"mmlu_{mmlu_split}_accuracy"] = np.mean(subject_scores)
        trainer.log(results)
        # trainer.data_collator.source_max_len = source_max_len
        return results

    return on_evaluate()


def run_evaluation(trainer: Trainer) -> Dict[str, float]:
    # Some patches to make stuff run
    trainer.args.remove_unused_columns = False
    # https://github.com/facebookresearch/llama-recipes/pull/196/files
    trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

    seq2seq_trainer = Seq2SeqTrainer(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        args=Seq2SeqTrainingArguments(
            **trainer.args.to_dict()
        ),
        data_collator=DataCollatorForCausalLM(
            tokenizer=trainer.tokenizer,
            source_max_len=2048,
            target_max_len=512,
            train_on_source=False,
            predict_with_generate=False,
        ),
    )

    # Run the evaluation
    return _run_evaluation(
        trainer=seq2seq_trainer,
        tokenizer=seq2seq_trainer.tokenizer)
