# LQ-LoRA: Low-rank plus Quantized Matrix Decomposition for Efficient Language Model Finetuning [[Paper](https://arxiv.org/abs/2311.12023)]

## Changelog
- 20231215: Uploaded artifacts.

## Artifacts
- Model checkpoint (and training logs) for LLaMA-2 7B with LQ-LoRA (2.75-bits, 64-rank, Fisher) [[link]](https://huggingface.co/hanguo/lq-lora/tree/main/output_c4wiki_lpq_20231022_ranks64_7b_c4_2.75)
- Model checkpoint (and training logs) for LLaMA-2 70B with LQ-LoRA (2.75-bits, 64-rank, Fisher) [[link]](https://huggingface.co/hanguo/lq-lora/tree/main/output_c4wiki_lpq_20231022_ranks64_70b_c4_2.75)
- Pre-computed ILP data for LLaMA-2 7B [[link]](https://huggingface.co/hanguo/lq-lora/tree/main/ilp_data_fp32)
- Pre-computed ILP data for LLaMA-2 70B [[link]](https://huggingface.co/hanguo/lq-lora/tree/main/ilp_data_bf16)
- Fisher Information for LLaMA-2 7B [[link]](https://huggingface.co/hanguo/lq-lora/tree/main/fisher_dict_fp32)
- Fisher Information for LLaMA-2 70B -> file over the size limit, please contact us!

## Installation

1. Clone the repo
```bash
git clone https://github.com/HanGuo97/lq-lora.git
cd lq-lora
```

2. Create Docker image (optional)
```bash
# Using BuiltKit
DOCKER_BUILDKIT=1 docker build \
    -t lqlora \
    -f Dockerfile \
    .

docker run -ti --rm \
    --gpus all \
    -p 28888:8888 \
    --shm-size=2g \
    lqlora \
    bash -c "cd main/ && jupyter-lab --ip=0.0.0.0 --allow-root"
```

3. Install dependencies
```bash
bash scripts/setup.sh
```

**Note**: Some of the codebase relies on PyTorch>=2.1.

## Usages

### Downloading Data for Quantization

After downloading the files, please update `FILE_NAMES_DICT` in `models/allocation_utils` accordingly.

### Applying Quantization

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import lora_utils

data = "c4"         # applying data-aware quantization
budget = "2.75"     # target bits
model_size = "70b"  # 7b or 70b

# Loads the base model (to CPU)
model = AutoModelForCausalLM.from_pretrained(
    f"meta-llama/Llama-2-{model_size}-hf")

# Adds LoRA components, etc
model = lora_utils.prepare_model_for_lora(
    model=model,
    num_ranks=64,
    lora_alpha=16,
    lora_dropout=0.0,
    use_gradient_checkpointing=True)

# Applies LQ-LoRA to the model.
lora_utils.transform_lora_layers(
    lpq=True,
    model=model,
    model_name=f"llama-2-{model_size}/lpq-64/{data},budget={budget}",
    device="cuda")
```

### Saving Quantized Models

Note that HuggingFace's PEFT library only saves the adapter parameters. Since LQ-LoRA additionally changes the base model parameters, we need to save the entire weights of the model.

```python
state_dict = model.state_dict()
file_name = os.path.join(
    output_dir,
    "full_model.pth")
torch.save(state_dict, file_name)
```

### Loading Quantized Models

```python
# No need to apply `transform_lora_layers` because
# these will be loaded from the checkpoint.
model = lora_utils.prepare_model_for_lora(
    model=model,
    num_ranks=64,
    lora_alpha=16,
    lora_dropout=0.0,
    use_gradient_checkpointing=True,
    checkpoint_dir=checkpoint_dir)  # -> enter the path to the checkpoint directory
```


## Todos
- [X] Upload the artifacts
- [ ] We use a legacy version of the (de)quantizaton implementation. We will update the code to use the latest version of the (de)quantization implementation.


## Acknowledgement

This code reuses components from several libraries including [QLoRA](https://github.com/artidoro/qlora) and [OmniQuant](https://github.com/OpenGVLab/OmniQuant).
