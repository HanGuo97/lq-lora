size="70b"
index=4

for hf_quantization_method in "gptq-3bit" "gptq-4bit"
do

echo "GPU ${index}: hf_quantization_method=${hf_quantization_method}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_oasst1.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --output_dir output_oasst1_lora_20230925_ranks64_${size}_${hf_quantization_method} \
    \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --bf16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing True \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to "wandb" \
    \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_config "lora-gptq" \
    --hf_quantization_method ${hf_quantization_method} &

index=$(($index+1))
sleep 60m

done


# --------------------------------------------------------------------------------



size="7b"
index=0

for lora_model_name in "nf3" "nf4"
do

echo "GPU ${index}: lora_model_name=${lora_model_name}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_oasst1.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --output_dir output_oasst1_lora_20230905_ranks64_${size}_${lora_model_name} \
    \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --bf16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing False \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to "wandb" \
    \
    --lora_r 64 \
    --lora_alpha 16 \
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

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_oasst1.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --output_dir output_oasst1_lora_20230905_ranks64_${size}_${data}_${budget} \
    \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --bf16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing False \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to "wandb" \
    \
    --lora_r 64 \
    --lora_alpha 16 \
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

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_oasst1.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --output_dir output_oasst1_lpq_20230905_ranks64_${size}_${data}_${budget} \
    \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --bf16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing False \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to "wandb" \
    \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_model_name "llama-2-${size}/lpq-64/${data},budget=${budget}" \
    --lora_dropout 0.0 \
    --lora_config "lora-lpq" &

index=$(($index+1))

done

done



# --------------------------------------------------------------------------------


size="7b"

echo "size=${size}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=6000 run_oasst1.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --output_dir output_oasst1_dense_20230927_${size} \
    \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --bf16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing False \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 2e-5 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to "wandb" \
    \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' &


# --------------------------------------------------------------------------------


size="7b"
index=4
checkpoint_dir="/export/share2/experiments/20231025/f39a6c538c5a/output_c4wiki_lpq_20231022_ranks64_7b_c4_2.75/"

echo "GPU ${index}"

WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_oasst1.py \
    --model_name_or_path /export/share3/experiments/20230731/llama-2/Llama-2-${size}-hf \
    --output_dir output_oasst1_continued_20231106_ranks64_${size} \
    \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --bf16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing False \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1875 \
    --eval_steps 187 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to "wandb" \
    \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_config ${checkpoint_dir}
