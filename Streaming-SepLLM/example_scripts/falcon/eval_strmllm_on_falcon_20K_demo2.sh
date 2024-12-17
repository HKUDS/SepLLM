CUDA_VISIBLE_DEVICES=0  python ../../main/evaluate_streaming_inputs_perplexity.py \
    --model_name_or_path tiiuae/falcon-40b\
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
    --output_dir ../../outputs/demo/xxx   2>&1 | tee ../../logs/demo/xxx.log

