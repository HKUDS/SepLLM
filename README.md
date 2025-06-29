<!-- <div align="center">

# **SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator**

### An Easy-to-Use Native Sparse Attention Baseline Method
---

</div> -->

<div align="center">
<h1 align="center">
  <strong>🚀 SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator</strong>
</h1>
<h3 align="center">✨ An Easy-to-Use <u><strong>Native Sparse Attention</strong></u> Baseline Method</h3>
<h4 align="center">
    <a href="https://sepllm.github.io" target="_blank">
      <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg" 
           alt="GitHub" 
           width="20" 
           height="20"
           style="vertical-align: middle; margin-right: 8px;"/>
      sepllm.github.io
    </a>
</h4>

---

</div>


Large Language Models (LLMs) have exhibited exceptional performance across a spectrum of natural language processing tasks. However, their substantial sizes pose considerable challenges, particularly in computational demands and inference speed, due to their quadratic complexity. In this work, we have identified a key pattern: certain seemingly meaningless separator tokens (i.e., punctuations) contribute disproportionately to attention scores compared to semantically meaningful tokens. This observation suggests that information of the segments between these separator tokens can be effectively condensed into the separator tokens themselves without significant information loss. Guided by this insight, we introduce SepLLM, a plug-and-play framework that accelerates inference by compressing these segments and eliminating redundant tokens. Additionally, we implement efficient kernels for training acceleration. Experimental results across training-free, training-from-scratch, and post-training settings demonstrate SepLLM's effectiveness. Notably, using the Llama-3-8B backbone, SepLLM achieves over 50% reduction in KV cache on the GSM8K-CoT benchmark while maintaining comparable performance. Furthermore, in streaming settings, SepLLM effectively processes sequences of up to 4 million tokens or more while maintaining consistent language modeling capabilities.


![image](https://hackmd.io/_uploads/r1POJoR4yg.png)



# News
![image](https://github.com/user-attachments/assets/54677ee5-85ad-4020-b39f-f3d8b34a7243)

- :star2: [2025/06] We are working on integrating `SepCache` into HuggingFace's [transformers](https://github.com/huggingface/transformers). Stay tuned! :rocket::rocket::rocket:
- :star2: [2025/06] `SepCache` is released, which is an efficient, portable, and easy-to-use *Cache* class for [transformers](https://github.com/huggingface/transformers). 
- :star2: [2025/06] SepLLM's trained [checkpoint samples](https://huggingface.co/Gausson/models) have been uploaded to [HuggingFace](https://huggingface.co/Gausson/models). :rocket::rocket::rocket:
- :star: [2025/06] More features have already been supported by the SepLLM code repository, including *BiPE* ([arXiv:2401.16421](https://arxiv.org/abs/2401.16421)), *Self-Adjust Softmax* ([arXiv:2502.18277](https://arxiv.org/abs/2502.18277)), *FixLLM*, *etc*.
- :star: [2025/06] SepLLM's [slides and videos](https://icml.cc/virtual/2025/poster/45536) are uploaded.
- :star: [2025/06] SepLLM's [camera-ready paper](https://icml.cc/virtual/2025/poster/45536) is released.
- :star2: [2025/05] **[SepLLM](https://icml.cc/virtual/2025/poster/45536) has been accepted to [ICML 2025](https://icml.cc/virtual/2025/poster/45536).** :rocket::rocket::rocket:
- :star: [2024/12] More exciting features are being developed. Stay tuned!
- :star: [2024/12] SepLLM's code has been released. Our codebase supports efficient multi-node distributed training with accelerated attention module *Sep-Attention* and also includes numerous existing Fusion Operators to accelerate the training process, such as *fused rope* ([Su et al., 2023](https://arxiv.org/abs/2104.09864)), *fused layer norm*, *etc*.

# Attention Please!
**Please pay extra attention to your usage and experimental scenarios, and choose the appropriate code subdirectory accordingly** (*i.e.*, `TrainingFree-SepLLM`, `Training-SepLLM`, `Streaming-SepLLM`). Some researchers have mistakenly used the `Streaming-SepLLM` folder's code for general training-free tasks (*e.g.*, `GSM8K_CoT`, `MMLU`, *etc.*), **which is incorrect**. The `Streaming-SepLLM` branch requires "Positional Encoding Shifting" like [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/), whereas general training-free tasks do not, as the context length and generation length required by such general tasks usually do not exceed the maximum length (`max_position_embeddings`) pre-trained by the model. Besides, there are other detailed differences, which can be found in the code. Due to the above reasons, we refer to `Streaming-SepLLM` as the "**Tailored** Streaming Design" in the [paper](https://arxiv.org/abs/2412.12094) to distinguish it from the "Fundamental Design." (Although we have made these two settings (`TrainingFree-SepLLM`, `Streaming-SepLLM`) compatible in [`SepCache`](#33-sepcache).)

In addition, to achieve optimal performance on downstream tasks, training from scratch is required (to ensure consistency between training and inference). However, for many downstream tasks, the training-free setting can also deliver quite good performance.


# 1. Overview
- [1. Overview](#1-overview)
- [2. Infinite-Length Streaming Tests](#2-infinite-length-streaming-tests)
  - [2.1 Usage](#21-usage)
- [3. Training-Free Tests](#3-training-free-tests)
  - [3.1 Environment Setup](#31-environment-setup)
  - [3.2 Quick-Start Usage](#32-quick-start-usage)
    - [3.2.1 Related Source Code Files](#321-related-source-code-files)
    - [3.2.2 Sample Usage](#322-sample-usage)
  - [3.3 SepCache](#33-sepcache)
    - [3.3.1 Related Source Code Files](#331-related-source-code-files)
    - [3.3.2 Basic Usage](#332-basic-usage)
      - [3.3.2.1 Initialization](#3321-initialization)
      - [3.3.2.2 Frequently-Used Parameters](#3322-frequently-used-parameters)
      - [3.3.2.3 Update Function](#3323-update-function)
      - [3.3.2.4 Summary](#3324-summary)
    - [3.3.3 Advanced Usage](#333-advanced-usage)
      - [3.3.3.1 Positional Encoding Shifting](#3331-positional-encoding-shifting)
      - [3.3.3.2 Other Advanced Parameters](#3332-other-advanced-parameters)
      - [3.3.3.3 Combined Use of Advanced Parameters](#3333-combined-use-of-advanced-parameters)
- [4. Training](#4-training)
  - [4.1 Data Preparation](#41-data-preparation)
  - [4.2 Environment Setup](#42-environment-setup)
    - [4.2.1 Conda](#421-conda)
    - [4.2.2 PDSH](#422-pdsh)
  - [4.3 Quick-Start with Sample Checkpoints on HuggingFace](#43-quick-start-with-sample-checkpoints-on-huggingface)
  - [4.4 Training by Yourself](#44-training-by-yourself)
    - [4.4.1 Basic Usage](#441-basic-usage)
    - [4.4.2 Advanced Usage](#442-advanced-usage)
      - [4.4.2.1 Distributed Training](#4421-distributed-training)
      - [4.4.2.2 SepLLM Configuration](#4422-sepllm-configuration)
      - [4.4.2.3 StreamingLLM Configuration](#4423-streamingllm-configuration)
      - [4.4.2.4 Self-Adjust (SA) Softmax Configuration](#4424-self-adjust-sa-softmax-configuration)
      - [4.4.2.5 Vanilla Full Attention](#4425-vanilla-full-attention)
      - [4.4.2.6 Other Custom Settings](#4426-other-custom-settings)
      - [4.4.2.7 Important Note](#4427-important-note)
  - [4.5 After-Training Evaluation](#45-after-training-evaluation)
- [5. Citation](#5-citation)

Here we provide an overview of our code repository. Some noteworthy sections are marked with an asterisk (*). Please note that the implementation of `SepCache` and the key code for the **Training-Free part** are not displayed here, as these codes are packaged in the wheel file `./package/transformers-4.38.0.post1+sepllm-py3-none-any.whl`. We will explain how to read, use, and modify these parts of the code in the corresponding sections.

We use `conda` (*e.g.*, Anaconda, Miniconda) to manage the **independent experimental environments** for different experiments.
```
SepLLM
├── package
│   ├── DeeperSpeed
│   │   ├── sm80_old_versions
│   │   │   └── ...
│   │   ├── sm90_old_versions
│   │   │   └── ...
│   │   ├── sm90_new_versions
│   │   │   └── cuda12.1-torch2.5.1-python3.10
│   │   │       └── deepspeed-0.12.4+6d097bec-cp310-cp310-linux_x86_64.whl
│   ├── DeepSpeed
│   │   └── cuda12.1-torch2.5.1-python3.10
│   │      └── deepspeed-0.14.5+unknown-cp310-cp310-linux_x86_64.whl
│   └── transformers-4.38.0.post1+sepllm-py3-none-any.whl
├── Streaming-SepLLM
│   ├── data
│   │   └── pg19
│   ├── debug
│   │   └── ...
│   ├── eval_sepllm_on_llama3_20K_demo1.1.sh
│   ├── ...
│   ├── example_scripts
│   │   ├── cal_avg_ppl.py
│   │   ├── falcon
│   │   │   └── ...
│   │   ├── gpt_neox
│   │   │   └── ...
│   │   ├── llama3
│   │   │   ├── eval_sepllm_on_llama3_20K_demo1.1.sh
│   │   │   ├── ...
│   │   │   ├── pg19
│   │   │   │   └── ...
│   │   │   └── wikitext
│   │   │       └── ...
│   ├── logs
│   │   └── demo
│   │       ├── llama3_8b_sepllm_len20480_ca800_loc256_sep64_init4_pg19_demo.log
│   │       └── ...
│   ├── main
│   │   └── evaluate_streaming_inputs_perplexity.py
│   ├── outputs
│   │   └── demo
│   │       ├── llama3_8b_sepllm_len20480_ca800_loc256_sep64_init4_pg19_demo.log
│   │       └── ...
│   ├── sepllm_kv_cache
│   │   ├── enable_pos_shifting.py
│   │   ├── kv_cache_manager.py
│   │   ├── pos_shift
│   │   │   ├── modify_llama.py
│   │   │   └── ...
│   │   └── utils.py
│   ├── setup.py
│   └── ...
│
├── TrainingFree-SepLLM
│   ├── Llama3_8B_Instruct_SepLLM_a3_n256_gsm8k_cot_eager.sh
│   ├── Llama3_8B_Instruct_SepLLM_gsm8k_cot_SepCache_a4_s128_w256_c512_with_flash_atten2.sh
│   ├── ...
│   ├── Llama3_trnfree_eval_logs
│   │   ├── sepllm_a3_n256_llama3_8B_inst_gsm8k_cot_eager.log
│   │   └── ...
│   └── Llama3_trnfree_sepllm_configs
│       ├── llama3_sepllm_a3_n256.yml
│       └── ...
└── Training-SepLLM
    ├── deepy.py
    ├── downstream_evaluation
    │   ├── eval_logs
    │   │   ├── pythia-160m-deduped-n128-SepLLM.log
    │   │   └── ...
    │   └── eval_scripts
    │       ├── eval_pythia-160m-deduped-n128-SepLLM.sh
    │       └── ...
    ├── eval.py
    ├── eval_tasks
    │   ├── eval_adapter.py
    │   └── ...
    ├── flex_attention.py -> /path/to/miniconda3/envs/py310_cu121_torch251_env/lib/python3.10/site-packages/torch/nn/attention/flex_attention.py
    ├── generate.py
    ├── LICENSE
    ├── logs
    │   └── example.log
    ├── MANIFEST.in
    ├── megatron
    │   ├── checkpointing.py
    │   ├── data
    │   │   ├── data_utils.py
    │   │   ├── helpers.cpp
    │   │   └── ...
    │   ├── fused_kernels
    │   │   ├── build
    │   │   │   ├── build.ninja
    │   │   │   ├── fused_rotary_positional_embedding.so
    │   │   │   └── ...
    │   │   ├── fused_rotary_positional_embedding.cpp
    │   │   ├── setup.py
    │   │   └── ...
    │   ├── gradient_noise_scale
    │   │   ├── gradient_noise_scale.py
    │   │   └── ...
    │   ├── initialize.py
    │   ├── __init__.py
    │   ├── learning_rates.py
    │   ├── logging.py
    │   ├── model
    │   │   ├── activations.py
    │   │   ├── fused_bias_dropout.py
    │   │   ├── fused_layer_norm.py
    │   │   ├── fused_rope.py
    │   │   ├── fused_softmax.py
    │   │   ├── gmlp.py
    │   │   ├── gpt2_model.py
    │   │   ├── init_functions.py
    │   │   ├── mamba
    │   │   │   └── ...
    │   │   ├── megablocks_utils.py
    │   │   ├── norms.py
    │   │   ├── positional_embeddings.py
    │   │   ├── rwkv
    │   │   │   └── ...
    │   │   ├── **sepllm_forward_input.py**
    │   │   ├── transformer.py
    │   │   ├── utils.py
    │   │   └── word_embeddings.py
    │   ├── mpu
    │   │   ├── cross_entropy.py
    │   │   ├── data.py
    │   │   ├── utils.py
    │   │   └── ...
    │   ├── mup_substitute.py
    │   ├── neox_arguments
    │   │   ├── arguments.py
    │   │   ├── deepspeed_args.py
    │   │   ├── neox_args.py
    │   │   └── ...
    │   ├── optimizers.py
    │   ├── **sepllm_attention.py**
    │   ├── text_generation_utils.py
    │   ├── tokenizer
    │   │   ├── tokenizer.py
    │   │   ├── train_tokenizer.py
    │   │   └── ...
    │   ├── training.py
    │   └── utils.py
├── prepare_data.py
├── requirements
│   └── ...
├── sample_configs
│   ├── 20B_tokenizer.json
│   ├── hostfile
│   ├── sepllm-160m-on-pythia-with-pile_deduped-n64-NOkernel.yml
│   └── ...
├── tests
│   └── ...
├── tools
│   ├── ckpts
│   │   ├── convert2HF.sh
│   │   ├── convert_neox_to_hf.py
│   │   └── ...
│   ├── datasets
│   │   └── ...
│   ├── general_bash
│   │   └── ...
│   ├── sepllm
│   │   └── get_separator_ids.py
│   └── ...
├── training_examples
│   ├── example160m_n64
│   └── ...
└── train.py

115 directories, 560 files
```


# 2. Infinite-Length Streaming Tests
Our long streaming evaluation is following [StreamingLLM.](https://github.com/mit-han-lab/streaming-llm/)

Take a quick look at the running process (recording of 20K tokens' generation based on Llama-3 - frame extraction and accelerated playback version).

https://github.com/user-attachments/assets/feb8b6e3-b523-40eb-beef-cad80a7db44b

Please see the full version of the video recording for generating 20K tokens below. You can also refer to the video cover (thumbnail) below for a summary of the running results.
<div align=center>
    <a href=https://sepllm.github.io><img src="https://github.com/user-attachments/assets/bce57c38-f1ea-49ae-850c-9a403ee552a8",  width=880 height=550>
</div>

<!-- <center>
    <img src="./Streaming-SepLLM/demo_video/small_demo.gif", width=800 height=500>
</center> -->

## 2.1 Usage 
To facilitate comparison with [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/), we have extended the [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/) codebase. Therefore, when conducting experiments in the streaming setting, we need to use the same runtime environment as [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/), such as `python==3.8`, `transformers==4.33.0`, `torch==2.1.0+cu121`, *etc*.
```bash
# Set conda environment
conda create -yn streaming_sepllm python=3.8
conda activate streaming_sepllm 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 # we use torch==2.1.0+cu121 for streaming test.
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

# Set Streaming-SepLLM
cd ./your_workspace
git clone https://github.com/HKUDS/SepLLM.git
cd ./SepLLM/Streaming-SepLLM
python setup.py develop
```
And to evaluate `Streaming-SepLLM`, you can follow this example:
```bash
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
    --split test \
    --output_dir ./outputs/xxx   2>&1 | tee ./logs/demo/xxx.log
```
Streaming setting typically involves positional encoding (PE) shifting similar to [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/). PE shifting means focusing on positions within the cache rather than those in the original text. To enable this feature:
- Set `--enable_pos_shift` to `True`.

To run tests for the vanilla full attention model, please set as follows:
- Set `--enable_kv_cache_manager` to `False`.
- Set `--enable_SepLLM` to `False` and set `enable_StreamingLLM` to `False`.
- Set `--enable_pos_shift` to `False`.

To run tests for the `SepLLM` or `StreamingLLM` architecture models, please set as follows:
- Set `--enable_kv_cache_manager` to `True`.
- Set `--enable_SepLLM` to `True` **OR** `--enable_StreamingLLM` to `True`. Only one `True` among these two parameters.

When `--enable_StreamingLLM` is set to `True`, certain parameters, such as `--sep_cache_size`, will not take effect and must satisfy the condition: `cache_size == init_cache_size + local_size`.

You can see many other examples under `./SepLLM/Streaming-SepLLM/example_scripts/`, including `SepLLM`, `StreamingLLM`, and `Full-Attention` on different backbone models of various sizes for generation tests of different lengths under various settings.



# 3. Training-Free Tests
In many scenarios, it is unnecessary or impractical to train models from scratch due to limited computational resources. However, in such cases, `SepLLM` can also demonstrate strong training-free performance. Based on `SDPA` or `eager` attention, we provide a simple and effective training-free implementation of `SepLLM` using a mask-based approach. While the mask-based method cannot really reduce the KV cache, it provides a convenient way for researchers to explore and analyze the behavior of attention mechanisms in LLMs.

Additionally, we introduce `SepCache`, an efficient, portable, and easy-to-use cache class for transformers. SepCache can significantly reduce KV cache, lowering GPU memory usage, improving throughput (*e.g.*, by increasing batch size), and reducing inference time. When using `flash_attention_2`, `SepCache` is employed by default to store past keys/values. This setup demonstrates how to use `SepCache` effectively and how it integrates with commonly used `flash_attention_2`.

All the above implementations are not directly included in this repository but are packaged in the wheel file `./package/transformers-4.38.0.post1+sepllm-py3-none-any.whl`. To explore, use, or modify the training-free implementation, you must first install this wheel package.

We have **ONLY** adapted the `Llama 3` series models (as well as our own `sepllm_gpt_neox` model, which will be introduced in detail later) into the `SepLLM` architecture. Since `Llama` (and `GPT_NeoX`) are among the most commonly used open-source models, we use them as examples to demonstrate how a pre-trained model can be transformed into the `SepLLM` architecture in a training-free manner. `SepLLM/TrainingFree-SepLLM/Llama3_trnfree_sepllm_configs` directory contains various SepLLM training-free configuration files for Llama-3 models (including `SepLLM`, `Vanilla`, `StreamingLLM`, `FixLLM`, *etc*.).

## 3.1 Environment Setup
```bash
# Create conda environment
conda env create -n trainingfree_sepllm python=3.10
conda activate trainingfree_sepllm
pip install flash-attn==2.7.2.post1 lm_eval==0.4.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121


# Download repo.
cd ./your_workspace
git clone https://github.com/HKUDS/SepLLM.git


# Install transformers of our version
cd ./SepLLM
pip install ./package/transformers-4.38.0.post1+sepllm-py3-none-any.whl # Required
cd ./TrainingFree-SepLLM
# Replace `/path/to/your_conda_directory/envs/trainingfree_sepllm` with your actual directory of your conda env.
ln -s /path/to/your_conda_directory/envs/trainingfree_sepllm/lib/python3.10/site-packages/transformers ./transformers # Create symbolic link
```
We have now created a conda environment named `trainingfree_sepllm`, where we have installed our released `transformers` package (**required**). Additionally, we have created **a symbolic link** to the source code of the installed `transformers` package under the `TrainingFree-SepLLM` directory, making it easier to read and modify the source code used for execution. Whenever we need to read or modify the `transformers` code, we can simply access it via `SepLLM/TrainingFree-SepLLM/transformers`.

## 3.2 Quick-Start Usage
### 3.2.1 Related Source Code Files
Please ensure that `SepLLM/package/transformers-4.38.0.post1+sepllm-py3-none-any.whl` has been installed in your conda env `trainingfree_sepllm` and a symbolic link to `transformers` has been created under `SepLLM/TrainingFree-SepLLM/` directory.

<!-- Please refer to the following locations to read and learn about the code for using SepLLM (especia). -->
Please refer to the following locations to read and learn about the code for using SepLLM (specifically for mask-based SepLLM with `sdpa` and `eager` attention, using the `Llama` model as an example **only**).
- `SepLLM/TrainingFree-SepLLM/transformers/models/llama/modeling_llama.py`
- `SepLLM/TrainingFree-SepLLM/transformers/models/llama/sepllm_attention.py`
- `SepLLM/TrainingFree-SepLLM/transformers/models/llama/sepllm_forward_input.py`


### 3.2.2 Sample Usage
We demonstrate how to adapt an LLM from the `Llama-3` series into the `SepLLM` architecture and run test tasks in a training-free manner. You can follow the example below to directly run a mask-based `SepLLM` using either the `eager` or `sdpa` attention mechanism. We use `GSM8K_CoT` as the example for evaluation and use `Llama-3` as an example to show how to do the adaptation.

```bash
# `sepllm_config` is necessary for 'sdpa' or 'eager' attention.
#  To obtain more accurate KV retention ratios, it's recommended to set the `batch size` to `1` to avoid the impact of padding tokens on the statistics.
lm_eval --model hf \
	--model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,attn_implementation=eager,sepllm_config=./Llama3_trnfree_sepllm_configs/llama3_sepllm_a3_n256.yml \
	--tasks    gsm8k_cot  \
    --device cuda:0\
	--batch_size 70 2>&1 | tee ./Llama3_trnfree_eval_logs/sepllm_a3_n256_llama3_8B_inst_gsm8k_cot_eager.log
```
Under the directory `SepLLM/TrainingFree-SepLLM/`, there are numerous example scripts for training-free experiments, including `SepLLM`, `Vanilla`, `StreamingLLM`, `FixLLM`, *etc*. You can conduct experiments to compare them (`FixLLM` is described in detail in `Appendix I. Fixed-Interval Variant` of our [paper](https://arxiv.org/abs/2412.12094)). And `SepLLM/TrainingFree-SepLLM/Llama3_trnfree_sepllm_configs` directory contains various SepLLM training-free configuration files for Llama-3 models (including `SepLLM`, `Vanilla`, `StreamingLLM`, `FixLLM`, *etc*.).

## 3.3 SepCache 
### 3.3.1 Related Source Code Files
Please refer to the following locations to read and learn about the `SepCache`-related code for using training-free `SepLLM` (in our example, `SepCache` is only invoked when using `flash_attention_2`. See the explanation in [`3.3.2 Basic Usage`](#332-basic-usage) for details. We use the `Llama` model as the example **only**).
- `SepLLM/TrainingFree-SepLLM/transformers/models/llama/modeling_llama.py`
- `SepLLM/TrainingFree-SepLLM/transformers/models/cache_utils.py`   ## `SepCache` code in it.

For example, you can learn how to initialize a `SepCache` object around `line 1070` in the `modeling_llama.py` file, and how to call the `update` function of a `SepCache` between lines around `487` and `507`.


### 3.3.2 Basic Usage
You can directly run the following commands, or simply execute the script `SepLLM/TrainingFree-SepLLM/Llama3_8B_Instruct_SepLLM_gsm8k_cot_SepCache_a4_s128_w256_c512_with_flash_atten2.sh` to learn how to use `SepCache`. Note that in our example, `SepCache` needs to be used in combination with `flash_attention_2`. This combination is necessary in our case because our goal is to demonstrate both the usage of `SepCache` and its integration with the commonly used `flash_attention_2`. However, this combined usage is not mandatory. Once you learn how to use `SepCache`, you'll find that you can easily integrate it with other attention methods.

**Note: In our example, the specific parameters for `SepCache` initialization are directly written on around `line 1070` in the `SepLLM/TrainingFree-SepLLM/transformers/models/llama/modeling_llama.py` file, not on the following script.**

```bash
# `sepllm_config` is unnecessary and not useful when using 'flash_attention_2' since 'flash_attention_2' is used together with `SepCache`. 
#  To obtain more accurate KV retention ratios, it's recommended to set the `batch size` to `1` to avoid the impact of padding tokens on the statistics.
#  You should pass your preferred settings as arguments at the point where `SepCache` is initialized (i.e., by calling `__init__` or `from_legacy_cache` (deprecated) function of `SepCache`)
lm_eval --model hf \
	--model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,attn_implementation=flash_attention_2 \
	--tasks    gsm8k_cot  \
    --device cuda:0\
	--batch_size 80 2>&1 | tee ./Llama3_trnfree_eval_logs/sepllm_llama3_8B_inst_gsm8k_cot_SepCache_a4_s128_w256_c512_with_flash_atten2.log
```

#### 3.3.2.1 Initialization
Now let's break down the basic usage of `SepCache`. First, you need to initialize an object of the `SepCache` class to serve as `past_key_values`. The basic initialization method is as follows. Here, `separator_token_ids: List[int]` and `PADDING_ID: int` must be provided unless `model_type` is specified (which must be one of our supported model types, such as "llama"). In that case, `separator_token_ids` and `PADDING_ID` will be automatically filled in.
```python
past_key_values = SepCache(         
        ## For SepLLM                                
        init_cache_size = 4,        
        sep_cache_size = 128,
        local_size=256, 
        cache_size=512,                            
        USE_MAX_SEP_CACHE = True,                        
        # separator_token_ids: List[int] = None, ## required for initialization if `model_type` is not provided.
        # PADDING_ID: int = None, ## required for initialization if `model_type` is not provided.

        ### For positional encoding shifting
        APPLY_PE_SHIFT = False,
        APPLY_PES_INSIDE = False,                        
        pe_dim = self.head_dim, ## The number of dims for positional encoding. Typically, just set the `head_dim` to this.
        max_position_embeddings = self.max_position_embeddings,                                    
        
        ## For basic transformer architecture
        k_seq_dim=2, ## The dimension for seq_len in key tensors
        v_seq_dim=2, ## The dimension for seq_len in value tensors
        layer_num = len(self.layers), ## required for initialization; it's the number of layers

        model_type = 'llama',  ## The model type for running the example. choose from ['llama', 'pythia','falcon'].                                                                        
       )
```
You can also use the `SepCache.from_legacy_cache()` function to create an object. Its parameters are the same as those of `__init__()`, with the only difference being that an additional `past_key_values` parameter needs to be specified. However, `from_legacy_cache()` is already deprecated, so when using it, the `past_key_values` parameter must be `None`. Therefore, in essence, `SepCache.from_legacy_cache()` and `SepCache()` are functionally the same.

**Important Note: In practice, no need to do positional encoding (PE) shifting like [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/) if the actual length does not exceed the pretrained max PE length (which applies to most downstream tasks.) . So, for most basic usages, just set `APPLY_PE_SHIFT=False` (`False` is also the default setting) and `APPLY_PES_INSIDE=False` for initialization.**

#### 3.3.2.2 Frequently-Used Parameters

Below, we provide explanations and examples for the most commonly used parameters when initializing `SepCache`.
```
SepCache stores the Key and Value states as lists of tensors, two lists for each layer. The expected shape for each tensor is
`[batch_size, num_heads, seq_len, head_dim]`.

Frequently-Used Parameters:

    `init_cache_size: Union[int, List]`:
        The maximum number of KVs to be stored for initial tokens.
        In the paper, the hyperparameter `a` is an abbreviated alias for `init_cache_size`.                
            
    `sep_cache_size: Union[int, List]`:
        The maximum number of KVs to be stored for separator tokens.
        In the paper, the hyperparameter `s` is an abbreviated alias for `sep_cache_size`.

    `local_size: Union[int, List]`: 
        The maximum number of KVs to be stored for local tokens (i.e., sliding window).
        In the paper, the hyperparameter `w` is an abbreviated alias for `local_size`.

    `cache_size: Union[int, List]`:    
        The maximum number of KVs to be stored for all the tokens, i.e., the size for the whole KV cache.  
        In the paper, the hyperparameter `c` is an abbreviated alias for `cache_size`.

    Concerning these four parameters above:
        When a list is passed (its length must be `layer_num`), it represents different values for each layer. 
        When an integer is passed, it means the setting is the same for all layers.
    
    
    `USE_MAX_SEP_CACHE: bool`: 
        If True, it means we only keep at most `sep_cache_size` seperators' KVs.  
        If the number exceeds this limit, older separators' KVs will be discarded, keeping only the most recent `sep_cache_size` KVs. 
        In the paper, the hyperparameter `s` is an abbreviated alias for `sep_cache_size`.
      
    `separator_token_ids: List[int]`:
        The token ids of the separator tokens for the current model's tokenizer.            
        We have some examples, such as the Llama-3 series models, where setting `model_type='llama'` allows you 
            to skip setting `separator_token_ids` and `PADDING_ID` (SepCache will auto-fill them).

    `PADDING_ID: int`:
        The token id of the padding token. You can just set `PADDING_ID` to the id of "<|endoftext|>" token of the tokenizer for the pretrained model.  
```
Important Note: 
- When `cache_size` and `local_size` are set to infinity (i.e., sufficiently large positive integers), and `USE_MAX_SEP_CACHE` is `False`, `SepCache` degenerates into a regular Cache. 
- You must always ensure that `init_cache_size` + `sep_cache_size` + `local_size` + `left_padding_offset` < `cache_size`. Here, `left_padding_offset` denotes the number of padding tokens in the record with the largest left paddings within a runtime batch. `left_padding_offset` can only be determined at runtime.        
- To guarantee the above inequality always holds during runtime, when setting, you can intentionally create a sufficient margin between both sides of the following inequality:
        `init_cache_size` + `sep_cache_size` + `local_size`  < `cache_size`, i.e., `a`+`s`+`w`<`c` in the [SepLLM paper - ICML 2025](https://arxiv.org/abs/2412.12094) to leave room for `left_padding_offset`.  

**More Important Note: In practice, no need to do positional encoding (PE) shifting like [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/) if the actual length does not exceed the pretrained max PE length (which applies to most downstream tasks.) . So, for most basic usages, just set `APPLY_PE_SHIFT=False` (`False` is also the default setting) and `APPLY_PES_INSIDE=False` for initialization.**


#### 3.3.2.3 Update Function
After initialization, another key point to note is that when using the `update` function of `SepCache` to update the **keys/values** and the **past token IDs** (which is necessary in SepCache), the current `input_ids` must also be provided.
```python
key_states, value_states = past_key_values.update(                
          key_states = key_states,
          value_states = value_states,    
          input_ids = input_ids,  ## required
          layer_idx = layer_idx,     
          PREFILLING_FLAG = q_len > 1, ## `q_len` is the sequence length of the current `query_states`
          )
```

#### 3.3.2.4 Summary
Here's a summary of the basic usage of SepCache:
```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, SepCache
>>> import torch
>>> from huggingface_hub import login
>>> login("hf_xxxXXXxxx")


>>> def to_cuda(a_dict: dict) -> dict:
>>>    new_dict = {}    
>>>    for k,v in a_dict.items():
>>>        if isinstance(v, torch.Tensor):
>>>            new_dict[k] = v.cuda()
>>>        else:
>>>            new_dict[k] = v
>>>    return new_dict

>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", attn_implementation="flash_attention_2", device_map="cuda:0")
>>> model.bfloat16().cuda()
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
>>> inputs = tokenizer(text="My name is Llama 3", return_tensors="pt")
>>> inputs = to_cuda(inputs)
>>> # Prepare a cache and pass it to model's forward; `layer_num` is the number of layers for the pretrained model.
>>> past_key_values = SepCache(init_cache_size=4, sep_cache_size=128, local_size=256, cache_size=512, layer_num=32, USE_MAX_SEP_CACHE=True, model_type='llama')
>>> # `separator_token_ids` and `PADDING_ID` must also be provided if you are not using `model_type='llama'` like this demo.
>>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
>>> outputs.past_key_values # access SepCache filled with keys/values
SepCache()

---------------------------------  
>>> ## When using the `update` function of SepCache to update the keys/values and the past token ids (necessary in SepCache), the current `input_ids` must also be provided.        
>>> key_states, value_states = past_key_values.update(                
            key_states = key_states,
            value_states = value_states,    
            input_ids = input_ids, ## required
            layer_idx = layer_idx,     
            PREFILLING_FLAG = q_len > 1, ## `q_len` is the sequence length of the current `query_states`
            )
```

### 3.3.3 Advanced Usage

#### 3.3.3.1 Positional Encoding Shifting

Advanced usage involves positional encoding (PE) shifting similar to [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/). PE shifting means SepLLM focuses on positions within the cache rather than those in the original text. To enable this feature:
- Set `APPLY_PE_SHIFT=True` when initializing `SepCache` object.
- By default, `APPLY_PES_INSIDE=True`, which means `SepCache` handles PE shifting internally.
```python
past_key_values = SepCache(         
        ## For SepLLM                                
        init_cache_size = 4,        
        sep_cache_size = 128,
        local_size=256, 
        cache_size=512,                            
        USE_MAX_SEP_CACHE = True,                        
        # separator_token_ids: List[int] = None, ## required for initialization if `model_type` is not provided.
        # PADDING_ID: int = None, ## required for initialization if `model_type` is not provided.

        ### For positional encoding shifting
        APPLY_PE_SHIFT = True, # For PE's shifting
        APPLY_PES_INSIDE = True, # For PE's shifting internally                   
        pe_dim = self.head_dim, ## The number of dims for positional encoding. Typically, just set the `head_dim` to this.
        max_position_embeddings = self.max_position_embeddings,                                    
        
        ## For basic transformer architecture
        k_seq_dim=2, ## The dimension for seq_len in key tensors
        v_seq_dim=2, ## The dimension for seq_len in value tensors
        layer_num = len(self.layers), ## required for initialization; it's the number of layers

        model_type = 'llama',  ## The model type for running the example. choose from ['llama', 'pythia','falcon'].                                                                        
       )
```
When `APPLY_PE_SHIFT=True`, two additional requirements must be noted when calling the `update` function of `SepCache`:
- `query_states` must also be passed to the `update` function. It will be updated and returned by the function.
- `position_ids` must also be provided. It is a Tensor of shape `[batch_size, seq_len]` during prefilling , and shape `[batch_size, 1]` during auto-regressive decoding phase, with `dtype=int`.

```python
key_states, value_states, query_states = past_key_values.update(                
    key_states = key_states,
    value_states = value_states,
    query_states = query_states, # additionally required
    input_ids = input_ids, 
    layer_idx = layer_idx, 
    position_ids = position_ids, # additionally required       
    PREFILLING_FLAG = q_len > 1,  ## `q_len` is the sequence length of the current `query_states`
    cache_kwargs = cache_kwargs   ## can be `None` if `APPLY_PES_INSIDE=True`
    )
```

Additionally, when `APPLY_PE_SHIFT=True` and `APPLY_PES_INSIDE=False`, it means you need to externally provide the sinusoidal matrices (`sin/cos` and optional `sin_q/cos_q` for `query_states`) required for positional encoding (PE) shifting. These should be passed via the `cache_kwargs` parameter (a `dict` type) in the `update` function. 
```python
### At least the shifted `sin` and `cos` should be properly provided (not `None`).
cache_kwargs = {"sin": sin, "cos": cos, "cos_q": cos_q, "sin_q": sin_q, "partial_rotation_size": None }
```
Here, `partial_rotation_size` means that only `partial_rotation_size` slices along certain dimension need to be shifted (i.e., [0, 1, ..., `partial_rotation_size-1`] slices along certain dimension). If `partial_rotation_size=None` (by default), it means all the slices along the dimension apply. The `partial_rotation_size` must always be passed through `cache_kwargs`, and it only takes effect when `APPLY_PE_SHIFT=True`.

#### 3.3.3.2 Other Advanced Parameters
When initializing `SepCache`, there are also some advanced parameters that can be configured, as shown below.
- `SEP_ACCUMULATION`:  If `True` (by default), it means we will try to accumulate all the KVs for seperators. If `False`, only the `new_sep_kv` compressed from the `past_win_kv` will be kept (see functions `__init__` and `compress_kv_cache_and_tokids_layer_wise` in `SepCache`). Simply put, when `SEP_ACCUMULATION=True`, we accumulate all KVs of separators compressed from the **Past Window Cache** into the **Separator Cache** until it's full. At that point, older separators' KVs are discarded to keep the size of **Separator Cache** within the limit `self.sep_cache_size`, retaining only the newer ones. When `SEP_ACCUMULATION=False`, only the newly compressed separators' KVs from the **Past Window Cache** are kept into **Separator Cache**, while all previously stored separators' KVs in the **Separator Cache** are discarded.
-  `USE_MAX_SEP_CACHE`: If `True`, it means we only keep at most `sep_cache_size` seperators' KVs.  If the number exceeds this limit, older separators' KVs will be discarded, keeping only the most recent `sep_cache_size` separators' KVs. Thus, `self.cache_size` also remains fixed at the preset value.
If `False` (by default), `self.sep_cache_size` will grow indefinitely as separators accumulate, causing `self.cache_size` to expand indefinitely as well. In the paper, the hyperparameter `s` is an abbreviated alias for `sep_cache_size`.
- `SEP_PADDING_IN_BATCH`: If `True`, it means that `SepCache` will pad separator tokens in other records to be aligned with the record with the most separators in a batch. If `False`, it means that `SepCache` will truncate older separator tokens in other records to be aligned with the record with the fewest separators in a batch. `False` by default.


#### 3.3.3.3 Combined Use of Advanced Parameters
If `SEP_ACCUMULATION=True` and `USE_MAX_SEP_CACHE=False`, as the number of input tokens increases, the number of separators' KVs in the KV cache will also accumulate endlessly, and `self.cache_size` and `self.sep_cache_size` will also be infinitely expanded (no longer fixed).

When `SEP_PADDING_IN_BATCH=True` is used in combination with `USE_MAX_SEP_CACHE=False` and `SEP_ACCUMULATION=True`, the KV cache will accumulate indefinitely, and since `SEP_PADDING_IN_BATCH=True`, the KVs of all seen separators will be retained (rather than being truncated).



# 4. Training

![image](https://hackmd.io/_uploads/r18jZD47Jg.png)


**Note: You must ensure that all configured paths involved in [`4. Training`](#4-training) are readable and writable for all nodes in the computer cluster.** Therefore, we recommend configuring a shared file system in the computing cluster that allows all nodes to have reading and writing access to. This eliminates the need to copy all source code files, configuration files, data, *etc.*, to each node in the cluster, avoiding unnecessary trouble. 

**Furthermore, you need to ensure that the source code files, configuration files, data, and *etc.* accessed by each node in the cluster are consistent.**


## 4.1 Data Preparation

We use the same training data as the [Pythia](https://github.com/EleutherAI/pythia) project, namely the open-source [deduped Pile](https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps) dataset, with the entire training process involving approximately 300 billion tokens. You can refer to the [Pythia](https://github.com/EleutherAI/pythia) project to download and prepare your training data, or refer to the summary of operations below.
```bash
# Download SepLLM code.
git clone https://github.com/HKUDS/SepLLM.git
cd SepLLM/Training-SepLLM/data_preparation

# Download the deduped Pile dataset.
git lfs clone https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps

# Optionally, to ensure against corrupt files
python utils/checksum_shards.py

python utils/unshard_memmap.py --input_file ./pythia_deduped_pile_idxmaps/pile_0.87_deduped_text_document-00000-of-00082.bin --num_shards 83 --output_dir ./pythia_pile_idxmaps/

# The correct sha256 for the full file is 0cd548efd15974d5cca78f9baddbd59220ca675535dcfc0c350087c79f504693
# This can be checked with sha256sum ./pythia_pile_idxmaps/*
```
You can store the dataset on a large hard drive or a shared file system of a distributed cluster, making it convenient for subsequent training on the distributed cluster's computing resources.

**Note: You must ensure that all configured paths involved in [`4. Training`](#4-training) are readable and writable for all nodes in the computer cluster.** Therefore, we recommend configuring a shared file system in the computing cluster that allows all nodes to have reading and writing access to. This eliminates the need to copy all source code files, configuration files, data, *etc.*, to each node in the cluster, avoiding unnecessary trouble. 

**Furthermore, you need to ensure that the source code files, configuration files, data, and *etc.* accessed by each node in the cluster are consistent.**


## 4.2 Environment Setup
### 4.2.1 Conda
We recommend using `torch 2.5.1` with `CUDA 12.1`. Alternatively, you can use `torch 2.1.0` with `CUDA 12.1`, but while this setup allows training to complete, some training features will be unavailable.

You need to install [`​​DeepSpeed`](https://www.deepspeed.ai/tutorials/advanced-install/)​​ or [`​​DeeperSpeed`](https://github.com/EleutherAI/DeeperSpeed)​​. We have prepared pre-built wheel packages for different environments (located in the `SepLLM/package/` directory), which you can install directly or customize based on your hardware and software setup following https://www.deepspeed.ai/tutorials/advanced-install/.

We recommend installing the relatively new versions under:
- `SepLLM/package/DeepSpeed/cuda12.1-torch2.5.1-python3.10/`

**OR**
- `SepLLM/package/DeeperSpeed/sm90_new_versions/cuda12.1-torch2.5.1-python3.10/`

Additionally, you need to install:
- `SepLLM/package/transformers-4.38.0.post1+sepllm-py3-none-any.whl`
- `lm_eval==0.4.0` for training.

```
package
├── DeeperSpeed
│   ├── sm80_old_versions
│   │   ├── cuda11.7-torch1.13-python3.8
│   │   └── cuda12.1-torch2.1-python3.8
│   ├── sm90_new_versions
│   │   └── cuda12.1-torch2.5.1-python3.10
│   └── sm90_old_versions
│       ├── cuda11.3-torch1.10.1-python3.8
│       ├── cuda11.3-torch1.11-python3.8
│       ├── cuda11.8-torch2.1-python3.8
│       ├── cuda11.8-torch2.4-python3.8
│       └── cuda12.1-torch2.1-python3.8
├── DeepSpeed
│   └── cuda12.1-torch2.5.1-python3.10
│       └── deepspeed-0.14.5+unknown-cp310-cp310-linux_x86_64.whl
└── transformers-4.38.0.post1+sepllm-py3-none-any.whl
```

```bash
conda env create -n training_sepllm python=3.10
conda activate training_sepllm 

cd SepLLM
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install ./package/transformers-4.38.0.post1+sepllm-py3-none-any.whl # Required
# Replace `/path/to/your_conda_directory/envs/training_sepllm` with your actual directory of your conda env.
ln -s /path/to/your_conda_directory/envs/training_sepllm/lib/python3.10/site-packages/transformers ./Training-SepLLM/downstream_evaluation/transformers # Create symbolic link

pip install ./package/DeepSpeed/cuda12.1-torch2.5.1-python3.10/deepspeed-0.14.5+unknown-cp310-cp310-linux_x86_64.whl # Recommended
pip install lm_eval==0.4.0
pip install -r ./Training-SepLLM/requirements/requirements_cuda12.1_torch2.5.1.txt
```
The dependency packages in the `SepLLM/Training-SepLLM/requirements/` directory are primarily inherited from the ​​[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) project. After setting up the conda environment, you can build some fused operators as follows:
```
cd SepLLM/Training-SepLLM/
python ./megatron/fused_kernels/setup.py install # optional if not using fused kernels
```
*Note: If you want to use the Sep-Attention module, please make sure your Pytorch>=2.5.0. And set "USE_SEP_ATTN_KERNEL_ACCELERATOR=True" in your training config file.*


Before starting training, please remember to use `Deep(er)Speed`'s `ds_report` command to check if it has been properly installed. If so, you will see an output like the one below. It is recommended to include as many fused operators as possible when installing `Deep(er)Speed` to accelerate training. However, this depends on your software and hardware environment, as adding operators may sometimes fail (compilation failure).

```bash
(training_sepllm)$ ds_report
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
async_io ............... [YES] ...... [OKAY]
fused_adam ............. [YES] ...... [OKAY]
cpu_adam ............... [YES] ...... [OKAY]
cpu_adagrad ............ [YES] ...... [OKAY]
cpu_lion ............... [YES] ...... [OKAY]
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
evoformer_attn ......... [NO] ....... [NO]
 [WARNING]  FP Quantizer is using an untested triton version (3.1.0), only 2.3.0 and 2.3.1 are known to be compatible with these kernels
fp_quantizer ........... [NO] ....... [NO]
fused_lamb ............. [YES] ...... [OKAY]
fused_lion ............. [YES] ...... [OKAY]
inference_core_ops ..... [YES] ...... [OKAY]
cutlass_ops ............ [YES] ...... [OKAY]
transformer_inference .. [YES] ...... [OKAY]
quantizer .............. [YES] ...... [OKAY]
ragged_device_ops ...... [YES] ...... [OKAY]
ragged_ops ............. [NO] ....... [OKAY]
random_ltd ............. [YES] ...... [OKAY]
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.5
 [WARNING]  using untested triton version (3.1.0), only 1.0.0 is known to be compatible
sparse_attn ............ [NO] ....... [NO]
spatial_inference ...... [NO] ....... [OKAY]
transformer ............ [YES] ...... [OKAY]
stochastic_transformer . [YES] ...... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/home/txiao/miniconda3/envs/py39_cu121_torch251_new2/lib/python3.10/site-packages/torch']
torch version .................... 2.5.1+cu121
deepspeed install path ........... ['/home/txiao/miniconda3/envs/py39_cu121_torch251_new2/lib/python3.10/site-packages/deepspeed']
deepspeed info ................... 0.14.5+unknown, unknown, unknown
torch cuda version ............... 12.1
torch hip version ................ None
nvcc version ..................... 12.1
deepspeed wheel compiled w. ...... torch 2.5, cuda 12.1
shared memory (/dev/shm) size .... 146.50 GB
```
### 4.2.2 PDSH

You must install the `pdsh` tool for distributed training, and you must ensure that each node in your computer cluster can connect to other nodes passwordlessly using the `ssh` tool. The `ninja` tool is also required for certain compilation processes.
Here are some reference materials about `​​pdsh`​​:
- https://linux.die.net/man/1/pdsh
- https://github.com/chaos/pdsh
- https://gist.github.com/mbbx6spp/c16b5438270be609619f83c462b148d5



## 4.3 Quick-Start with Sample Checkpoints on [HuggingFace](https://huggingface.co/Gausson/models)

<!-- ![image](https://github.com/user-attachments/assets/4a8807d3-6955-490e-b113-d13ac3aa6c27) -->
<div align=center>
    <a href=https://huggingface.co/Gausson/models><img src="https://github.com/user-attachments/assets/4a8807d3-6955-490e-b113-d13ac3aa6c27",  width=880 >
</div>



If you don't have time or computational resources to train from scratch, we have uploaded our pre-trained checkpoints to [Hugging Face](https://huggingface.co/Gausson/models) Hub. After setting up the `training_sepllm` conda environment, you can directly evaluate the checkpoints on downstream tasks.
- ​​Evaluation Scripts​​: Located in `SepLLM/Training-SepLLM/downstream_evaluation/eval_scripts/`
- Result Logs​​: Saved in `SepLLM/Training-SepLLM/downstream_evaluation/eval_logs/`
- HuggingFace Model's Code: Located in `SepLLM/Training-SepLLM/downstream_evaluation/transformers/models/sepllm_gpt_neox/`
```
downstream_evaluation
├──transformers
│   └── ...
├── eval_logs
│   ├── gpt-neox-125m-deduped-SA.log
│   ├── pythia-160m-deduped.log
│   ├── pythia-160m-deduped-n128-SepLLM.log
│   ├── pythia-160m-deduped-n64h-SepLLM.log
│   ├── pythia-160m-deduped-n64ht-SepLLM.log
│   ├── pythia-160m-deduped-n64-RoBiPE-SepLLM.log
│   ├── pythia-160m-deduped-n64-SepLLM.log
│   ├── pythia-160m-deduped-n64-StreamingLLM.log
│   └── pythia-160m-deduped-SepLLM.log
└── eval_scripts
    ├── eval_gpt-neox-125m-deduped-SA.sh
    ├── eval_pythia-160m-deduped-n128-SepLLM.sh
    ├── eval_pythia-160m-deduped-n64h-SepLLM.sh
    ├── eval_pythia-160m-deduped-n64ht-SepLLM.sh
    ├── eval_pythia-160m-deduped-n64-RoBiPE-SepLLM.sh
    ├── eval_pythia-160m-deduped-n64-SepLLM.sh
    ├── eval_pythia-160m-deduped-n64-StreamingLLM.sh
    ├── eval_pythia-160m-deduped-SepLLM.sh
    └── eval_pythia-160m-deduped.sh
```
We trained the `SepLLM` architecture model based on the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) backbone, which we refer to as the `sepllm_gpt_neox` architecture model. In the HuggingFace-format `transformers` package **we released**, the model architecture name and path are also `sepllm_gpt_neox`.
In the file `SepLLM/Training-SepLLM/downstream_evaluation/transformers/models/sepllm_gpt_neox/modeling_sepllm_gpt_neox.py`, classes such as `SepLLMGPTNeoXModel` and `SepLLMGPTNeoXForCausalLM` are defined. 

Here, we did not directly integrate `SepCache` into `sepllm_gpt_neox` because we want `sepllm_gpt_neox` to serve as a straightforward and simple mask-based `SepLLM` model for research exploration, rather than being directly used for downstream applications. Once you become familiar with `SepLLM`, integrating `SepCache` into `sepllm_gpt_neox` is a very simple task, and you can give it a try.
```
sepllm_gpt_neox
├── __init__.py
├── configuration_sepllm_gpt_neox.py
├── modeling_sepllm_gpt_neox.py
├── sepllm_attention.py
├── sepllm_forward_input.py
└── tokenization_gpt_neox_fast.py
```

Below we provide a sample script to evaluate a trained checkpoint. You can find additional example scripts in:
`SepLLM/Training-SepLLM/downstream_evaluation/eval_scripts/`, where we provide example scripts about our [`SepLLM`](https://arxiv.org/abs/2412.12094) models of various pretraining settings (*e.g.*, integrated with [`BiPE`](https://arxiv.org/abs/2401.16421)), [`StreamingLLM`](https://arxiv.org/abs/2309.17453) models, [`Self-Adjust Softmax (SA)`](https://arxiv.org/abs/2502.18277) models, [`Vanilla Pythia Models`](https://github.com/EleutherAI/pythia), *etc*.
```bash
CUDA_LAUNCH_BLOCKING=1
# You can set the `pretrained` field of `--model_args` to `/path/to/your_converted_hf_checkpoints` for your own trained and converted HF checkpoints
lm_eval --model hf \
	--model_args pretrained=Gausson/pythia-160m-deduped-SepLLM \
	--tasks  arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc,wikitext  \
	--num_fewshot 5 \
	--device cuda:0\
	--batch_size 32 2>&1 | tee ../eval_logs/pythia-160m-deduped-SepLLM.log
```

However, under typical circumstances, you should:
- Train your own `SepLLM` models​​ using our provided training code.
- ​​Convert the trained checkpoints​​ to Hugging Face format using our conversion script: `SepLLM/Training-SepLLM/tools/ckpts/convert2HF.sh`
- Evaluate the converted checkpoints​​ using the downstream test scripts mentioned above. When running evaluation scripts, specify your converted checkpoint path by setting the `pretrained` field in the `--model_args` parameter.

## 4.4 Training by Yourself

**Note: You must ensure that all configured paths involved in [`4. Training`](#4-training) are readable and writable for all nodes in the computer cluster.** Therefore, we recommend configuring a shared file system in the computing cluster that allows all nodes to have reading and writing access to. This eliminates the need to copy all source code files, configuration files, data, *etc.*, to each node in the cluster, avoiding unnecessary trouble. 

**Furthermore, you need to ensure that the source code files, configuration files, data, and *etc.* accessed by each node in the cluster are consistent.**




In our default training parameter settings, `"use_shared_fs"` is set to `True`. Its definition can be found in the data class `NeoXArgsTraining` in `SepLLM/Training-SepLLM/megatron/neox_arguments/neox_args.py` as follows (please appropriately set this parameter in your training YAML configuration file):
```python
    use_shared_fs: bool = True
    """
    Whether to use a shared filesystem for data loading. If False, local rank 0 on all nodes will preprocess the data,
    otherwise only global rank 0 will preprocess the data. This is implemented in SepLLM/Training-SepLLM/megatron/data/gpt2_dataset.py::_build_index_mappings.
    """
```

The key files and directories involved in training are as follows:
- `SepLLM/Training-SepLLM/sample_configs/` for sample YAML configuration files.
- `SepLLM/Training-SepLLM/training_examples/` for sample training scripts.
- `SepLLM/Training-SepLLM/megatron/model/sepllm_forward_input.py` for SepLLM.
- `SepLLM/Training-SepLLM/megatron/sepllm_attention.py` for SepLLM.
- `SepLLM/Training-SepLLM/megatron/neox_arguments/neox_args.py` for parameter definitions.
- `SepLLM/Training-SepLLM/megatron/model/transformer.py` for transformer.
- `SepLLM/Training-SepLLM/megatron/model/gpt2_model.py` for backbone model.
- `SepLLM/Training-SepLLM/megatron/training.py` for training process.





### 4.4.1 Basic Usage

You can start training just by:
```bash
cd SepLLM/Training-SepLLM/
python ./deepy.py ./train.py path/to/config.yml
```
We have prepared numerous training launch script examples in the directory `SepLLM/Training-SepLLM/training_examples/`, in which each example demonstrates different architectures or hyperparameter configurations, including:
- Training `SepLLM` models with various settings (*e.g.*, different hyperparameters and integration with [`BiPE`](https://arxiv.org/abs/2401.16421).)
- [`StreamingLLM`](https://arxiv.org/abs/2309.17453) models, [`Self-Adjust Softmax (SA)`](https://arxiv.org/abs/2502.18277) models, Vanilla Full Attention models, *etc*.
```
training_examples
├── example1.4b_n64
├── example1.4b_n64_kernel
├── example160m_n128
├── example160m_n128_kernel
├── example160m_n128_kernel_recompile
├── example160m_n64
├── example160m_n64H
├── example160m_n64H_kernel
├── example160m_n64H_kernel_recomplie
├── example160m_n64HT
├── example160m_n64HT_alibiBiPE
├── example160m_n64HT_kernel
├── example160m_n64HT_kernel_recompile
├── example160m_n64HT_kernel_recompile_alibiBiPE
├── example160m_n64HT_kernel_recompile_rotaryBiPE
├── example160m_n64HT_rotaryBiPE
├── example160m_n64_kernel
├── example160m_n64_kernel_recompile
├── streamingllm-160m
├── vanilla-1.4b-full-attention
└── vanilla-160m-full-attention
```
All corresponding YAML configuration files as well as the vocabulary file (`vocab-file`) and `hostfile`  can be found in the directory `SepLLM/Training-SepLLM/sample_configs/`
```
sample_configs
├── 20B_tokenizer.json
├── debug.yml
├── hostfile
├── sepllm-1.4b-on-pythia_32heads-with-pile_deduped-n64-kernel.yml
├── sepllm-1.4b-on-pythia_32heads_with-pile_deduped-n64-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n128HMT-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n128HT-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n128-kernel-recompile.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n128-kernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n128-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n256HMT-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n256-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64H-kernel_recompile.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64H-kernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HMT-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64H-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HT-kernel_recompile_alibiBiPE.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HT-kernel_recompile_rotaryBiPE.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HT-kernel_recompile.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HT-kernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HT-NOkernel_alibiBiPE.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HT-NOkernel_rotaryBiPE.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64HT-NOkernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64-kernel-recompile.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64-kernel.yml
├── sepllm-160m-on-pythia-with-pile_deduped-n64-NOkernel.yml
├── streamingllm-160m-n64.yml
├── vanilla-1.4b-32heads-full-attention.yml
└── vanilla-160m-full-attention.yml
```
When using YAML configuration files, you must modify:
- The training data input path
- The vocabulary file path (`vocab-file`)
Please update these paths in your YAML file according to the following format:
```yaml
"train-data-paths": ["/path/to/your_pythia_deduped_data/pile_0.87_deduped_text_document"],
"valid-data-paths": ["/path/to/your_pythia_deduped_data/pile_0.87_deduped_text_document"],
"test-data-paths":  ["/path/to/your_pythia_deduped_data/pile_0.87_deduped_text_document"],
"vocab-file": "/path/to/your_workspace/SepLLM/Training-SepLLM/sample_configs/20B_tokenizer.json",
# "hostfile": "/path/to/your_workspace/SepLLM/Training-SepLLM/sample_configs/hostfile",
```
After completing the data preparation steps in [`4.1 Data Preparation`](#41-data-preparation), you will obtain two data files: 
- `pile_0.87_deduped_text_document.bin` 
- `pile_0.87_deduped_text_document.idx`. 

You must​​ copy both files to `/path/to/your_pythia_deduped_data/` and ensure this directory is ​​accessible to all nodes​​ in your computer cluster (with both reading and writing permissions) and at runtime, the program may generate additional files (*e.g*., data shuffle files) in this directory. Besides, you just set `"vocab-file"` to the path to your `20B_tokenizer.json` under the directory `SepLLM/Training-SepLLM/sample_configs/`.

In addition, you can set the path to save your checkpoints using `"save"` field. The `"load"` field also allows you to resume training from the most recent checkpoint when training is terminated. The `"checkpoint-factor"` is used to set how often to save a checkpoint, based on the number of steps.

```yaml
"checkpoint-factor": 1000,
"save": "/path/to/save_and_load/your_checkpoints",
"load": "/path/to/save_and_load/your_checkpoints",
```
**Note: You must ensure that all paths mentioned here are readable and writable for all nodes in the computer cluster.**

### 4.4.2 Advanced Usage

#### 4.4.2.1 Distributed Training
You can set the path to the `"hostfile"` file in a YAML configuration file to specify the node information for the computer 
cluster used in distributed training.
```yaml
"hostfile": "/path/to/your_workspace/SepLLM/Training-SepLLM/sample_configs/hostfile",
```
The content of the `"hostfile"` should include the visible IP for each of your nodes and the number of GPUs available on each node. Please note that distributed training relies on the `pdsh` tool (see [`4.2.2 PDSH`](#422-pdsh)), and it is crucial to ensure that every node in the cluster can establish passwordless `SSH` connections to each other.
```
node1_ip slots=8
node2_ip slots=8
```
If you only have one node, you can choose to remove (or just comment out) the `"hostfile"` field in the YAML configuration file.


#### 4.4.2.2 SepLLM Configuration
In the file `SepLLM/Training-SepLLM/megatron/neox_arguments/neox_args.py`, you can find the `SepLLMArgs` data class. You can learn how to use these configuration parameters based on the comments provided within it. The numerous example YAML configuration files located under the directory `SepLLM/Training-SepLLM/sample_configs/` are also excellent resources for learning how to configure parameters related to `SepLLM`. Below, we provide the definition and comments of the `SepLLMArgs` data class.
```python
@dataclass
class SepLLMArgs(NeoXArgsTemplate):
    """
    Our SepLLM args when training
    """

    separator_token_ids: list = None
    """
    The token ids for the special tokens (i.e. separators):  ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'].
    Use [-1] to disable. 
    """
    
    PADDING_ID:  int = 0  # For pythia (GPT_NeoX)    
    """
    The id for padding token of Pythia (GPT_NeoX)
    """

    prefill_k: int = 0  ## NOT implemented yet; From old version: Deprecated          
    generate_k: int  = 0  ## NOT implemented yet; From old version: Deprecated
    """
    The max number of layers (excluded, layers: [0, prefill_k) or [0, generate_k) ) that use the original attention masks (upper triangular matrices) when prefilling and generating respectively. These two args are NOT implemented yet and deprecated.
    For now, put a large number (>=max_seq_len) for the corresponding layers in prefill_loc_win_size_list (or generate_win_loc_size_list) if you want to keep the entire layer's KV and attentions
    """

    prefill_local_window_size: int  = 256  
    """
    The local window size when training and prefilling.  KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.

    Only take effect when USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS=False. 
    """
    
    generate_local_window_size: int  = 256 
    """
    The local window size when generating. KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
    
    Only take effect when USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS=False. generate_local_window_size does not have any effect during the pretraining/prefilling phase.
    """

    USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS: bool = False
    """
    If True: the prefilling local window sizes for different self-attention layers are different.
    If True: should set 'prefill_loc_win_size_list', else: should set 'prefill_local_window_size'
    """

    USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS: bool = False 
    """
    If True: the generating local window sizes for different self-attention layers are different.
    If True: should set 'generate_win_loc_size_list', else: should set 'generate_local_window_size'  
    USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS does not have any effect during the pretraining/prefilling phase.
    """

    prefill_loc_win_size_list: list = None
    """
    The local window sizes for different self-attention layers when training (or prefilling). KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
    """

    generate_win_loc_size_list: list = None
    """
    The local window sizes for different self-attention layers when generating. KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
    generate_win_loc_size_list does not have any effect during the pretraining/prefilling phase.
    """

    init_tok_max_idx: int = 2 
    """
    The largest index for the kept initial tokens. E.g., if init_tok_max_idx==2, it means we keep 3 initial tokens (idx: 0,1,2)
    """

    ######################################There should be at most 1 True for the following 3 args ##############################################
    USE_ORIGINAL_FULL_ATTEN: bool = False  
    """
    Flag signal with the highest priority.  Train the Pythia model without any modification (standard full-attention version, i.e., standard upper triangular mask) if True.
    """

    streamingLLM: bool = False 
    """
    Train streamingLLM. Only takes effect when USE_ORIGINAL_FULL_ATTEN=False. 
    """

    USE_SEP_ATTN_KERNEL_ACCELERATOR: bool = True 
    """
    If True, use Sep_Attention module's kernel accelerator to accelerate the training process of SepLLM. If False (together with USE_ORIGINAL_FULL_ATTEN=False and streamingLLM=False), run plain SepLLM
    """
    ######################################There should be at most 1 True for the above 3 args ##############################################
    
    RECOMPILE_SEP_ATTN_KERNEL: bool = False 
    """
    False by default. If True, recompile the Sep_Attention kernels.  When set to True, it may require more GPU memory and provide a certain level of acceleration to the training process.
    """

    BATCH_ADAPTIVE_INIT_POS: bool = False 
    """
    If True: use the floating initial tokens' starting positions since when evaluating, LLM usually add paddings on the left side of the shorter seqs in a batch for alignment (i.e., left padding).
    
    Can be False when pretraining since the starting positions of initial tokens are at the beginning of each sequence in a batch for pretraining (i.e., right padding)
    """

    PRINT_KV_RATIO: bool = False 
    """
    If True, print the KV cache preservation ratio (especially for the released trained model during generating). When pretraining, it will also print the retention ratio for the computational complexity of calculating the attention map if it is set True
    """

    print_ratio_intervals: int = 8000
    """    
    Print the retention ratio for the computational complexity of calculating the attention map once after every 'print_ratio_intervals' forward passes (or print_ratio_intervals/gradient_accumulation_steps  iterations). It only takes effect when PRINT_KV_RATIO=True.    
    """

    USE_BiPE: bool = False 
    """
    False by default. If True (must also set pos_emb='rotary' or 'alibi'), use Bilevel Positional Encoding [He, Zhenyu, et al. "Two stones hit one bird: Bilevel positional encoding for better length extrapolation." arXiv preprint arXiv:2401.16421 (2024).].
    """

    BiPE_seps: list = None
    """
    The token ids of the seperator tokens for BiPE.  
    """
    
    ###################################### Read-only Hyperparameter ##############################################
    EXCLUDE_DIAGONAL: bool = True ## From old version: Deprecated
    """
    True by default and should always be True. When True, it means when we choose fixed window to process the prefilling mask, the diagonal elements in the prefilling mask could be set negative. When False: would keep the prefilling mask's diagonal positive
    """

    _DEFAULT_SPARSE_BLOCK_SIZE: int = 128
    """
    128 by default. It can also be set to align with the head dimension.
    """
```
We also provide many pre-trained `SepLLM` models, which you can find on [HuggingFace](https://huggingface.co/Gausson/models).

#### 4.4.2.3 StreamingLLM Configuration

You can train a [`StreamingLLM`](https://arxiv.org/abs/2309.17453) architecture model using our code repository, and we also provide a pre-trained `StreamingLLM` model on our [`HuggingFace`](https://huggingface.co/Gausson/pythia-160m-deduped-n64-StreamingLLM) platform for your reference and comparison. Note that the training of the `StreamingLLM` architecture model is mutually exclusive with the `SepLLM` architecture and the later-mentioned `Self-Adjust (SA) Softmax` architecture, meaning only one type of model can be independently trained at a time.

You can set the `"streamingLLM"` field to `True` in your YAML configuration file to train a `StreamingLLM` architecture model. We provide a reference configuration file at `SepLLM/Training-SepLLM/sample_configs/streamingllm-160m-n64.yml`.


#### 4.4.2.4 Self-Adjust (SA) Softmax Configuration

Our training code repository also supports training models based on [`Self-Adjust (SA) Softmax`](https://arxiv.org/abs/2502.18277) attention. Note that `SA` and `SepLLM` are two different model architectures, so only one of them (`SA` or `SepLLM`) can be trained independently at a time. They are incompatible and cannot be trained simultaneously. Therefore, when you want to train an `SA` model, many related parameters for `SepLLM` will not take effect. Below is the data class for `SA`-related parameters, `AdjustSoftmaxArgs`, and you can read the comments to understand it. This data class, `AdjustSoftmaxArgs`, is also defined in `SepLLM/Training-SepLLM/megatron/neox_arguments/neox_args.py`. Similarly, there are sample YAML configuration files for `SA` training available under `SepLLM/Training-SepLLM/sample_configs/`.
```python
@dataclass
class AdjustSoftmaxArgs(NeoXArgsTemplate):
    """
    Our Self-Adjusting Softmax attention args when training
    """

    ######################################There should be at most 1 True for the following 2 args ##############################################
    USE_SA_SOFTMAX: bool = False
    """
    False by default. If True, use Self-Adjusting Softmax Attention.
    """
    
    USE_SA_SOFTMAX_NO_DENO: bool = False  
    """
    False by default. If True, use Self-Adjusting Softmax Attention V2 : no denominator version.
    """
    ######################################There should be at most 1 True for the above 2 args ##################################################

    SA_Numerator_Bias: float = 0.  
    """
    The bias value added to the numerator term of Self-Adjusting Softmax Attention.
    """

    SA_Denominator_Bias: float = 0.0000000001  
    """
    The bias value added to the denominator term of Self-Adjusting Softmax Attention.
    """
```
We also provide pre-trained `SA` models, which you can find on [HuggingFace](https://huggingface.co/Gausson/models).


#### 4.4.2.5 Vanilla Full Attention 

You can, of course, also train a Vanilla Full Attention model by setting the `"USE_ORIGINAL_FULL_ATTEN"` field to `True` in your YAML configuration file. Doing so will disable all other settings related to `SepLLM`, `StreamingLLM`, and `SA`, leaving only the basic training settings effective. We provide a reference training configuration file at `SepLLM/Training-SepLLM/sample_configs/vanilla-160m-full-attention.yml`. 

The checkpoints obtained through this method, once converted to `HuggingFace` format as described in [`4.5 After-Training Evaluation`](#45-after-training-evaluation), will still use the model name `sepllm_gpt_neox` in `transformers` package we released. However, apart from the difference in name, it is essentially identical to the vanilla `gpt_neox` model.


#### 4.4.2.6 Other Custom Settings
The various other parameters involved in the training process are defined under `SepLLM/Training-SepLLM/megatron/neox_arguments/`. In addition to this, apart from referring to the YAML configuration file examples under `SepLLM/Training-SepLLM/sample_configs/`, you can also refer to the following materials, which come from the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) project and the [Pythia](https://github.com/EleutherAI/pythia) project.
- `https://github.com/EleutherAI/gpt-neox/blob/main/configs/README.md`
- `https://github.com/EleutherAI/pythia`
- `https://github.com/EleutherAI/gpt-neox`


#### 4.4.2.7 Important Note
In the training YAML configuration file, only **one** of the following fields:
- `"USE_ORIGINAL_FULL_ATTEN"` 
- `"streamingLLM"` 
- `"USE_SEP_ATTN_KERNEL_ACCELERATOR"`
- `"USE_SA_SOFTMAX"`
- `"USE_SA_SOFTMAX_NO_DENO"`

can be set to `True`, corresponding to a specific training mode. When all of them are set to `False`, the model will be trained using the plain `SepLLM` architecture (*i.e.*, pure mask-based `SepLLM` with no acceleration). Refer to the example configuration files and comments in the `SepLLM/Training-SepLLM/sample_configs/` directory to learn how to use them, which is not difficult.


For further interactions between parameters, please refer to the `SepLLMArgumentsChecker` class located in `SepLLM/Training-SepLLM/megatron/utils.py`. This class will also verify the validity of your parameter settings before training begins and provide appropriate prompts.


## 4.5 After-Training Evaluation
After completing the training, if you are using a distributed shared file system, you can directly use the script below to convert the checkpoints obtained from training into `HuggingFace` format checkpoints for downstream task testing.

If you are not using a distributed shared file system, you need to copy the relevant state files of the checkpoints saved on each node of the cluster to a single node, and then use the script below to perform the format conversion.
```bash
python convert_neox_to_hf.py --input_dir /path/to/your_model_checkpoints/global_stepXXX \
	--config_file /path/to/your_model_checkpoints/global_stepXXX/configs/your_configs.yml \
	--output_dir /path/to/your_converted_hf_checkpoints \
	--precision auto \
	--architecture sepllm_gpt_neox
```
The `convert_neox_to_hf.py` file is located in the `SepLLM/Training-SepLLM/tools/ckpts/` directory. You can refer to the script file `SepLLM/Training-SepLLM/tools/ckpts/convert2HF.sh` to execute this format conversion.

`sepllm_gpt_neox` is the model architecture name or model name in `HuggingFace` format. You must refer to [`4.2 Environment Setup`](#42-environment-setup) to set up the environment `training_sepllm` (one of the most important steps is installing the `transformers-4.38.0.post1+sepllm-py3-none-any.whl` package we provided) to support the `sepllm_gpt_neox` model type.

After completing the conversion, you can use [`lm_eval`](https://github.com/EleutherAI/lm-evaluation-harness) to test the model you trained. For details, please refer to [`4.3 Quick-Start with Sample Checkpoints on HuggingFace`](#43-quick-start-with-sample-checkpoints-on-huggingface). Below, we also provide a testing example. More testing samples and results can be found in the directory `SepLLM/Training-SepLLM/downstream_evaluation/`. Additionally, we have uploaded our trained examples to the [HuggingFace](https://huggingface.co/Gausson/models) platform for reference.


```bash
CUDA_LAUNCH_BLOCKING=1
# You can set the `pretrained` field of `--model_args` to `/path/to/your_converted_hf_checkpoints` for your own trained and converted HF checkpoints
lm_eval --model hf \
	--model_args pretrained=Gausson/pythia-160m-deduped-SepLLM \
	--tasks  arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc,wikitext  \
	--num_fewshot 5 \
	--device cuda:0\
	--batch_size 32 2>&1 | tee ../eval_logs/pythia-160m-deduped-SepLLM.log
```
The sample result for the above testing example is as follows, and you can find it in the following file: `SepLLM/Training-SepLLM/downstream_evaluation/eval_logs/pythia-160m-deduped-SepLLM.log`.
```
hf (pretrained=Gausson/pythia-160m-deduped-SepLLM), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 32
|    Tasks     |Version|Filter|n-shot|    Metric     |   | Value |   |Stderr|
|--------------|------:|------|-----:|---------------|---|------:|---|------|
|arc_challenge |      1|none  |     5|acc            |↑  | 0.2014|±  |0.0117|
|              |       |none  |     5|acc_norm       |↑  | 0.2346|±  |0.0124|
|arc_easy      |      1|none  |     5|acc            |↑  | 0.4731|±  |0.0102|
|              |       |none  |     5|acc_norm       |↑  | 0.4520|±  |0.0102|
|lambada_openai|      1|none  |     5|acc            |↑  | 0.3315|±  |0.0066|
|              |       |none  |     5|perplexity     |↓  |30.1605|±  |1.0128|
|logiqa        |      1|none  |     5|acc            |↑  | 0.2273|±  |0.0164|
|              |       |none  |     5|acc_norm       |↑  | 0.2857|±  |0.0177|
|piqa          |      1|none  |     5|acc            |↑  | 0.6464|±  |0.0112|
|              |       |none  |     5|acc_norm       |↑  | 0.6447|±  |0.0112|
|sciq          |      1|none  |     5|acc            |↑  | 0.8260|±  |0.0120|
|              |       |none  |     5|acc_norm       |↑  | 0.8150|±  |0.0123|
|wikitext      |      2|none  |     5|bits_per_byte  |↓  | 0.9207|±  |   N/A|
|              |       |none  |     5|byte_perplexity|↓  | 1.8931|±  |   N/A|
|              |       |none  |     5|word_perplexity|↓  |30.3488|±  |   N/A|
|winogrande    |      1|none  |     5|acc            |↑  | 0.5304|±  |0.0140|
|wsc           |      1|none  |     5|acc            |↑  | 0.3750|±  |0.0477|
```

Besides, the script below provides a method for testing the full attention vanilla official `Pythia` models (not the `SepLLM` architecture) for your reference and comparison. This testing script is saved in `SepLLM/Training-SepLLM/downstream_evaluation/eval_scripts/eval_pythia-160m-deduped.sh`, and a sample of the test results is available in `SepLLM/Training-SepLLM/downstream_evaluation/eval_logs/pythia-160m-deduped.log`. For more information about the vanilla `Pythia` models, please refer to https://github.com/EleutherAI/pythia.
```bash
CUDA_LAUNCH_BLOCKING=1
lm_eval --model hf \
	--model_args pretrained=EleutherAI/pythia-160m-deduped \
	--tasks  arc_challenge,arc_easy,lambada_openai,logiqa,piqa,sciq,winogrande,wsc,wikitext  \
	--num_fewshot 5 \
	--device cuda:0\
	--batch_size 32 2>&1 | tee ../eval_logs/pythia-160m-deduped.log
```

# 5. Citation
If you find our work helpful, please consider giving us a star :star: and citing our paper. We greatly appreciate your support 😄
```
@inproceedings{chen2025sepllm,
  title={{SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator}},
  author={Chen, Guoxuan and Shi, Han and Li, Jiawei and Gao, Yihang and Ren, Xiaozhe and Chen, Yimeng and Jiang, Xin and Li, Zhenguo and Liu, Weiyang and Huang, Chao},
  booktitle={Proceedings of the Forty-Second International Conference on Machine Learning (ICML)},
  year={2025},
  note={Also available at arXiv:2412.12094}
}
```
