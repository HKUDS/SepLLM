# `sepllm_config` is unnecessary and not useful when using 'flash_attention_2' since 'flash_attention_2' is used together with `SepCache`. 
#  To obtain more accurate KV retention ratios, it's recommended to set the `batch size` to `1` to avoid the impact of padding tokens on the statistics.
#  You should pass your preferred settings as arguments at the point where `SepCache` is initialized (i.e., by calling `__init__` or `from_legacy_cache` (deprecated) function of `SepCache`)
lm_eval --model hf \
	--model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,attn_implementation=flash_attention_2 \
	--tasks    gsm8k_cot  \
    --device cuda:0\
	--batch_size 80 2>&1 | tee ./Llama3_trnfree_eval_logs/sepllm_llama3_8B_inst_gsm8k_cot_SepCache_a4_s128_w256_c512_with_flash_atten2.log

