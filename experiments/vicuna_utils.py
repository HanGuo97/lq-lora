import os
import json
import torch
import click
import jsonlines
import shortuuid
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict
from peft import PeftModelForCausalLM
from transformers import PreTrainedTokenizerBase
from fastchat.model import get_conversation_template


def run_eval(
    model: PeftModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    model_id: str,
    question_file: str,
    answer_file: str,
    num_gpus: int,
) -> None:
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // num_gpus
    ans_jsons = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_jsons.extend(
            get_model_answers(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                question_jsons=ques_jsons[i : i + chunk_size],
            )
        )

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")


# @torch.inference_mode()
def get_model_answers(
    model: PeftModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    model_id: str,
    question_jsons: List,
) -> List[Dict]:

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        conv = get_conversation_template(model_id)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            inputs=torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {},
            }
        )
    return ans_jsons


def postprocess(base_dir: str, verbose: bool = False) -> None:
    wildcard = os.path.join(base_dir, "*.jsonl")
    for file_name in glob(wildcard):
        print(file_name)

        objs_processed = []
        with jsonlines.open(file_name) as reader:
            for obj in reader:
                obj_processed = deepcopy(obj)

                if "###" in obj["text"]:
                    obj_processed["text"] = obj["text"].split("###")[0]

                    if verbose is True:
                        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        click.secho(obj["text"], fg="blue")
                        print("--------------------------------")
                        click.secho(obj_processed["text"], fg="red")
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                objs_processed.append(obj_processed)

        with jsonlines.open(f"{file_name}.processed", mode="w") as writer:
            writer.write_all(objs_processed)
