import click
import torch
import random
from tqdm.auto import tqdm
from typing import Union, Dict
from datasets import load_dataset
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase)
from peft import PeftModelForCausalLM


def _check_tokenizer_type(tokenizer: PreTrainedTokenizerBase) -> None:
    if not isinstance(tokenizer, PreTrainedTokenizer):
        raise TypeError(
            f"Expected a `transformers.PreTrainedTokenizer` object, "
            f"but got a {type(tokenizer)} object instead.")
    if tokenizer.is_fast is not False:
        raise ValueError(
            f"Expected `tokenizer.is_fast` to be False, "
            f"but got {tokenizer.is_fast} instead.")


def get_wikitext2(tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
    _check_tokenizer_type(tokenizer)
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return testenc


def get_c4(tokenizer: PreTrainedTokenizerBase, seqlen: int) -> torch.Tensor:
    _check_tokenizer_type(tokenizer)
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation")

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return valenc 


def get_c4_new(tokenizer: PreTrainedTokenizerBase, seqlen: int) -> torch.Tensor:
    _check_tokenizer_type(tokenizer)
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation")

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return valenc


def get_loaders(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    seqlen: int = 2048,
) -> Union[BatchEncoding, torch.Tensor]:
    if "wikitext2" in dataset_name:
        return get_wikitext2(tokenizer=tokenizer)
    if "c4" in dataset_name:
        if "new" in dataset_name:
            return get_c4_new(tokenizer=tokenizer, seqlen=seqlen)
        return get_c4(tokenizer=tokenizer, seqlen=seqlen)
    raise ValueError


@torch.no_grad()
def legacy_evaluation(
    model: PeftModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    seqlen: int = 2048,
) -> Dict[str, float]:
    # Legacy GPTQ-style evaluation for consisteny with previous results
    click.secho(
        f"Setting `model.config.use_cache` "
        f"(was {model.config.use_cache}) to `False`",
        fg="blue")

    results = {}
    for dataset_name in ["wikitext2", "c4", "c4-new"]:
        testloader = get_loaders(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            seqlen=seqlen)
        if "c4" in dataset_name:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
            logits = model(batch).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        model.config.use_cache = use_cache
        results[dataset_name] = ppl.item()
    return results
