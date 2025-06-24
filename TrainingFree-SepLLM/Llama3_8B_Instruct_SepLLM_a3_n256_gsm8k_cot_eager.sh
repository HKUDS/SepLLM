# `sepllm_config` is necessary for 'sdpa' or 'eager' attention.
#  To obtain more accurate KV retention ratios, it's recommended to set the `batch size` to `1` to avoid the impact of padding tokens on the statistics.
lm_eval --model hf \
	--model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,attn_implementation=eager,sepllm_config=./Llama3_trnfree_sepllm_configs/llama3_sepllm_a3_n256.yml \
	--tasks    gsm8k_cot  \
    --device cuda:0\
	--batch_size 70 2>&1 | tee ./Llama3_trnfree_eval_logs/sepllm_a3_n256_llama3_8B_inst_gsm8k_cot_eager.log
