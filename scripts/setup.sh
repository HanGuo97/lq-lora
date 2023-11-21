pip install --upgrade pip
pip install -U -r requirements.txt

pip install -U optimum
pip install "fschat[model_worker,webui]"
pip install auto-gptq==0.4.2 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
pip install pytorch-quantization==2.1.3 --extra-index-url https://pypi.ngc.nvidia.com

wandb login
