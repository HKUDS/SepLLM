CUDA_VISIBLE_DEVICES=0  python ./main/evaluate_streaming_inputs_perplexity.py \
    --model_name_or_path  meta-llama/Meta-Llama-3-8B\
    --init_cache_size 4 \
    --sep_cache_size 64 \
    --local_size 256 \
    --cache_size 800 \
    --enable_kv_cache_manager True \
    --enable_SepLLM True \
    --enable_StreamingLLM False \
    --enable_pos_shift True \
    --num_samples 5000000 \
    --num_eval_tokens 20480 \
    --dataset_name pg19 \
    --task default \
    --split test\
    --output_dir ./outputs/demo/llama3_8b_sepllm_len20480_ca800_loc256_sep64_init4_pg19_demo   2>&1 | tee ./logs/demo/llama3_8b_sepllm_len20480_ca800_loc256_sep64_init4_pg19_demo.log


