#python ./tools/ckpts/convert_neox_to_hf.py --input_dir /path/to/model/global_stepXXX --config_file your_config.yml --output_dir hf_model/save/location --precision {auto,fp16,bf16,fp32} --architecture {neox,mistral,llama}

python ./ckpts/convert_neox_to_hf.py --input_dir /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_seg/pythia160m-deduped/scaled_masked_softmax_fusion/checkpoints_k0_w128/global_step23000 \
	--config_file /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_seg/pythia160m-deduped/scaled_masked_softmax_fusion/checkpoints_k0_w128/global_step23000/configs/pythia-160m-deduped_new_seg_k0_w128_NOtri.yml \
	--output_dir /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_seg/pythia160m-deduped/scaled_masked_softmax_fusion/checkpoints_k0_w128/hf_checkpoints/global_step23000 \
	--precision auto \
	--architecture neox
	
# /lustre/fast/fast/txiao/shihan/saves/gpt-neox-new_seg/pythia160m-deduped/checkpoints_k0_w128