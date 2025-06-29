CUDA_LAUNCH_BLOCKING=1
# You can set the `pretrained` field of `--model_args` to `/path/to/your_converted_hf_checkpoints` for your own trained and converted HF checkpoints
lm_eval --model hf \
	--model_args pretrained=Gausson/gpt-neox-125m-deduped-SA \
	--tasks  arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc,wikitext  \
	--num_fewshot 5 \
	--device cuda:0\
	--batch_size 32 2>&1 | tee ../eval_logs/gpt-neox-125m-deduped-SA.log
