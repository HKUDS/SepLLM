CUDA_VISIBLE_DEVICES=0  python ./main/evaluate_streaming_inputs_perplexity.py \
    --model_name_or_path  meta-llama/Meta-Llama-3-8B\
    --init_cache_size 4 \
    --local_size 796 \
    --cache_size 800 \
    --enable_kv_cache_manager True \
    --enable_SepLLM False \
    --enable_StreamingLLM True \
    --enable_pos_shift True \
    --num_samples 5000000 \
    --num_eval_tokens 20480 \
    --dataset_name pg19 \
    --task default \
    --split test\
    --output_dir ./outputs/demo/llama3_8b_streamingllm_len20480_ca800_loc796_init4_pg19_demo   2>&1 | tee ./logs/demo/llama3_8b_streamingllm_len20480_ca800_loc796_init4_pg19_demo.log


