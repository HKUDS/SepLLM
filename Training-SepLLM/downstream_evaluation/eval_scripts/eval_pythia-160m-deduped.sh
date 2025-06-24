CUDA_LAUNCH_BLOCKING=1
lm_eval --model hf \
	--model_args pretrained=EleutherAI/pythia-160m-deduped \
	--tasks  arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc,wikitext  \
	--num_fewshot 5 \
	--device cuda:0\
	--batch_size 32 2>&1 | tee ../eval_logs/pythia-160m-deduped.log
