size="7b"
index=0

for hf_quantization_method in "gptq-3bit" "gptq-4bit"
do

echo "GPU ${index}: hf_quantization_method=${hf_quantization_method}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_clm.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --dataset_name "c4" \
    --block_size 1024 \
    --bf16 True \
    --output_dir output_c4_lora_20230909_ranks64_${size}_${hf_quantization_method} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "steps" \
    --eval_steps 0.34 \
    --save_strategy "steps" \
    --save_steps 0.34 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --tf32 True \
    --low_cpu_mem_usage True \
    --lora_num_ranks 64 \
    --lora_dropout 0.0 \
    --lora_config "lora-gptq" \
    --hf_quantization_method ${hf_quantization_method} &

index=$(($index+1))

done


# --------------------------------------------------------------------------------


size="7b"
index=0

for lora_model_name in "nf3" "nf4"
do

echo "GPU ${index}: lora_model_name=${lora_model_name}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_clm.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --dataset_name "c4" \
    --block_size 1024 \
    --bf16 True \
    --output_dir output_c4_lora_20230909_ranks64_${size}_${lora_model_name} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "steps" \
    --eval_steps 0.34 \
    --save_strategy "steps" \
    --save_steps 0.34 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --tf32 True \
    --lora_num_ranks 64 \
    --lora_model_name ${lora_model_name} \
    --lora_dropout 0.0 \
    --lora_config "lora" &

index=$(($index+1))

done


# --------------------------------------------------------------------------------


size="7b"
index=0

for data in "None"
do

for budget in "4" "3.75" "3.5" "3.25" "3" "2.75" "2.5"
do

echo "GPU ${index}: data=${data}, budget=${budget}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_clm.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --dataset_name "c4" \
    --block_size 1024 \
    --bf16 True \
    --output_dir output_c4_lora_20230909_ranks64_${size}_${data}_${budget} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "steps" \
    --eval_steps 0.34 \
    --save_strategy "steps" \
    --save_steps 0.34 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --tf32 True \
    --lora_num_ranks 64 \
    --lora_model_name "llama-2-${size}/lora/${data},budget=${budget}" \
    --lora_dropout 0.0 \
    --lora_config "lora" &

index=$(($index+1))

done

done


# --------------------------------------------------------------------------------


size="7b"
index=0

for data in "None" "c4"
do

for budget in "4" "3.75" "3.5" "3.25" "3" "2.75" "2.5"
do

echo "GPU ${index}: data=${data}, budget=${budget}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_clm.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --dataset_name "c4" \
    --block_size 1024 \
    --bf16 True \
    --output_dir output_c4_lpq_20230909_ranks64_${size}_${data}_${budget} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "steps" \
    --eval_steps 0.34 \
    --save_strategy "steps" \
    --save_steps 0.34 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --tf32 True \
    --lora_num_ranks 64 \
    --lora_model_name "llama-2-${size}/lpq-64/${data},budget=${budget}" \
    --lora_dropout 0.0 \
    --lora_config "lora-lpq" &

index=$(($index+1))

done

done


# --------------------------------------------------------------------------------


index=0

for size in "7b"
do

echo "GPU ${index}: size=${size}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_clm.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --dataset_name "c4" \
    --block_size 1024 \
    --bf16 True \
    --output_dir output_c4_dense_20230909_${size} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "steps" \
    --eval_steps 0.34 \
    --save_strategy "steps" \
    --save_steps 0.34 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval \
    --tf32 True &

index=$(($index+1))

done


# --------------------------------------------------------------------------------


size="70b"
data="c4"
budget="2.75"

echo "GPU ${index}: data=${data}, budget=${budget}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=6000 run_clm.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --dataset_name "c4-wiki-large" \
    --block_size 2048 \
    --bf16 True \
    --output_dir output_c4wiki_lpq_20231022_ranks64_${size}_${data}_${budget} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "steps" \
    --eval_steps 0.34 \
    --save_strategy "steps" \
    --save_steps 0.34 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --tf32 True \
    --low_cpu_mem_usage True \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --ddp_timeout 7200 \
    --use_fast_tokenizer False \
    --lora_num_ranks 64 \
    --lora_model_name "llama-2-${size}/lpq-64/${data},budget=${budget}" \
    --lora_dropout 0.0 \
    --lora_config "lora-lpq"
