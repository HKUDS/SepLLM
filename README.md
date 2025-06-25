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


# Overview
Here we provide an overview of our code repository. Some noteworthy sections are marked with an asterisk (*). Please note that the implementation of `SepCache` and the key code for the **Training-Free part** are not displayed here, as these codes are packaged in the wheel file `./package/transformers-4.38.0.post1+sepllm-py3-none-any.whl`. We will explain how to read, use, and modify these parts of the code in the corresponding sections.

We use `conda` (*e.g.*, Anaconda, Miniconda) to manage the experimental environments for different experiments.
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


# Infinite-Length Streaming Tests
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

## Usage 
To facilitate comparison with [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/), we have extended the [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/) codebase. Therefore, when conducting experiments in the streaming setting, we need to use the same runtime environment as [StreamingLLM](https://github.com/mit-han-lab/streaming-llm/), such as `python==3.8`, `transformers==4.33.0`, `torch==2.1.0+cu121`, *etc*.
```
# Set conda environment
conda create -yn streaming_sepllm python=3.8
conda activate streaming_sepllm 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 # we use torch==2.1.0+cu121 for streaming test.
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

# Set Streaming-SepLLM
cd ./your_workplace
git clone https://github.com/HKUDS/SepLLM.git
cd ./SepLLM/Streaming-SepLLM
python setup.py develop
```
And to evaluate `Streaming-SepLLM`, you can follow this example:
```
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
    --output_dir ./outputs/xxx   2>&1 | tee ./logs/demo/xxx.log
```
You can see many other examples under `./SepLLM/Streaming-SepLLM/example_scripts/`, including `SepLLM`, `StreamingLLM`, and `Full-Attention` on different backbone models of various sizes for generation tests of different lengths.


# Training-Free Tests
In many scenarios, it is unnecessary or impractical to train models from scratch due to limited computational resources. However, in such cases, `SepLLM` can also demonstrate strong training-free performance. Based on `SDPA` or `eager` attention, we provide a simple and effective training-free implementation of `SepLLM` using a mask-based approach. While the mask-based method cannot really reduce the KV cache, it provides a convenient way for researchers to explore and analyze the behavior of attention mechanisms in LLMs.

Additionally, we introduce `SepCache`, an efficient, portable, and easy-to-use cache class for transformers. SepCache can significantly reduce KV cache, lowering GPU memory usage, improving throughput (*e.g.*, by increasing batch size), and reducing inference time. When using `flash_attention_2`, `SepCache` is employed by default to store past keys/values. This setup demonstrates how to use `SepCache` effectively and how it integrates with commonly used `flash_attention_2`.

All the above implementations are not directly included in this repository but are packaged in the wheel file `./package/transformers-4.38.0.post1+sepllm-py3-none-any.whl`. To explore, use, or modify the training-free implementation, you must first install this wheel package.

We have only adapted the `Llama 3` series models (as well as our own `sepllm_gpt_neox` model, which will be introduced in detail later) into the `SepLLM` architecture. Since `Llama` (and `GPT_NeoX`) are among the most commonly used open-source models, we use them as examples to demonstrate how a pre-trained model can be transformed into the `SepLLM` architecture in a training-free manner. `SepLLM/TrainingFree-SepLLM/Llama3_trnfree_sepllm_configs` directory contains various training-free configuration files for Llama-3 models (including `SepLLM`, `Vanilla`, `StreamingLLM`, `FixLLM`, *etc*.).

## Environment Setup
```
# Create conda environment
conda env create -n trainingfree_sepllm python=3.10
conda activate trainingfree_sepllm
pip install flash-attn==2.7.2.post1 lm_eval==0.4.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121


# Download repo.
cd ./your_workplace
git clone https://github.com/HKUDS/SepLLM.git


# Install transformers of our version
cd ./SepLLM
pip install ./package/transformers-4.38.0.post1+sepllm-py3-none-any.whl # Required
cd ./TrainingFree-SepLLM
# Replace `/path/to/your_conda_directory/envs/trainingfree_sepllm` with your actual directory of your conda env.
ln -s /path/to/your_conda_directory/envs/trainingfree_sepllm/lib/python3.10/site-packages/transformers ./transformers 
```
We have now created a conda environment named `trainingfree_sepllm`, where we have installed our released `transformers` package (required). Additionally, we have created a symbolic link to the source code of the installed `transformers` package under the `TrainingFree-SepLLM` directory, making it easier to read and modify the source code used for execution. Whenever we need to read or modify the `transformers` code, we can simply access it via `SepLLM/TrainingFree-SepLLM/transformers`.

## Quick-Start Usage
We demonstrate how to adapt an LLM from the `Llama-3` series into the `SepLLM` architecture and run test tasks in a training-free manner. You can follow the example below to directly run a mask-based `SepLLM` using either the `eager` or `sdpa` attention mechanism. We use `GSM8K_CoT` as the example for evaluation and use `Llama-3` as an example to show how to do the adaptation.

```
# `sepllm_config` is necessary for 'sdpa' or 'eager' attention.
#  To obtain more accurate KV retention ratios, it's recommended to set the `batch size` to `1` to avoid the impact of padding tokens on the statistics.
lm_eval --model hf \
	--model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,attn_implementation=eager,sepllm_config=./Llama3_trnfree_sepllm_configs/llama3_sepllm_a3_n256.yml \
	--tasks    gsm8k_cot  \
    --device cuda:0\
	--batch_size 70 2>&1 | tee ./Llama3_trnfree_eval_logs/sepllm_a3_n256_llama3_8B_inst_gsm8k_cot_eager.log
```
Under the directory `SepLLM/TrainingFree-SepLLM`, there are numerous example scripts for training-free experiments, including `SepLLM`, `Vanilla`, `StreamingLLM`, `FixLLM`, *etc*. You can conduct experiments to compare them. (`FixLLM` is described in detail in `Appendix I. Fixed-Interval Variant` of our [paper](https://arxiv.org/abs/2412.12094).)


# Training

You can install the required package in the requirements.txt. You are recommended to build a independent conda environment (or pyenv, etc.) to do this. Our code is based on the code framework [GPTNeoX](url=https://github.com/EleutherAI/gpt-neox).



![image](https://hackmd.io/_uploads/r18jZD47Jg.png)

## Usage

All the code corresponding to training is in the Training-SepLLM folder. If you want to use the fused operators, just run:
```
cd Training-SepLLM
pip install -r requirements/requirements.txt
python ./megatron/fused_kernels/setup.py install # optional if not using fused kernels
```
*Note: If you want to use the Sep-Attention module, please make sure your Pytorch>=2.5.0. And set "USE_SEP_ATTN_KERNEL_ACCELERATOR=True" in your training config file.*

You can start training by:
```
python ./deepy.py train.py [path/to/config.yml]
```
The sample configuration yml files are in ./Training-SepLLM/sample_configs.

### Parameter Settings for SepLLM Training

```
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
    
    Only take effect when USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS=False.
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
    """



    prefill_loc_win_size_list: list = None
    """
    The local window sizes for different self-attention layers when training (or prefilling). KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
    """

    generate_win_loc_size_list: list = None
    """
    The local window sizes for different self-attention layers when generating. KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
    """

    init_tok_max_idx: int = 2 
    """
    The largest index for the kept initial tokens. E.g., if init_tok_max_idx==2, it means we keep 3 initial tokens (idx: 0,1,2)
    """


    ######################################There should be at most 1 True for the following 3 args ##############################################
    USE_ORIGINAL_FULL_ATTEN: bool = False  
    """
    Flag signal with the highest priority.  Run the model without any modification (standard full-attention version, i.e., standard upper triangular mask) if True.
    """

    streamingLLM: bool = False 
    """
    Run streamingLLM. Only takes effect when USE_ORIGINAL_FULL_ATTEN=False. 
    """

    USE_SEP_ATTN_KERNEL_ACCELERATOR: bool = True 
    """
    If True, use Sep_Attention module to accelerate the training process of SepLLM
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
    If True, print the KV cache preservation ratio (especially for the released trained model during generating). When pretraining, it will print the retention ratio for the computational complexity of calculating the attention map if it is set True
    """

    print_ratio_intervals: int = 8000
    """    
    Print the retention ratio for the computational complexity of calculating the attention map once after every 'print_KV_intervals' forward passes (or print_KV_intervals/gradient_accumulation_steps  iterations). It only takes effect when PRINT_KV_RATIO=True.    
    """

    USE_BiPE: bool = False 
    """
    False by default. If True (must also set pos_emb='rotary' or 'alibi'), use Bilevel Positional Encoding.  [He, Zhenyu, et al. "Two stones hit one bird: Bilevel positional encoding for better length extrapolation." arXiv preprint arXiv:2401.16421 (2024).]
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
```

Remember to save your training process checkpoints, so that if the training is interrupted unexpectedly, you can resume the training. You can set the save dir in the config yml file.
```
  "save": "path/to/checkpoints",
  "load": "path/to/checkpoints",
```


After the training is completed, we can convert the training checkpoints to the Hugging Face format, so that we can test them on downstream tasks （e.g. using [lm_eval](url=https://github.com/EleutherAI/lm-evaluation-harness)）.

```
python ./tools/ckpts/convert_neox_to_hf.py --input_dir path/to/checkpoints/global_stepXXX --config_file your_config.yml --output_dir hf_model/save/dir
```
