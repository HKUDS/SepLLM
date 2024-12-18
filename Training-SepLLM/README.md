# Abstract
Large Language Models (LLMs) have exhibited exceptional performance across a spectrum of natural language processing tasks. However, their substantial sizes pose considerable challenges, particularly in computational demands and inference speed, due to their quadratic complexity. In this work, we have identified a key pattern: certain seemingly meaningless special tokens (i.e., separators) contribute disproportionately to attention scores compared to semantically meaningful tokens. This observation suggests that information of the segments between these separator tokens can be effectively condensed into the separator tokens themselves without significant information loss. Guided by this insight, we introduce SepLLM, a plug-and-play framework that accelerates inference by compressing these segments and eliminating redundant tokens. Additionally, we implement efficient kernels for training acceleration. Experimental results across training-free, training-from-scratch, and post-training settings demonstrate SepLLM's effectiveness. Notably, using the Llama-3-8B backbone, SepLLM achieves over 50% reduction in KV cache on the GSM8K-CoT benchmark while maintaining comparable performance. Furthermore, in streaming settings, SepLLM effectively processes sequences of up to 4 million tokens or more while maintaining consistent language modeling capabilities.


![image](https://hackmd.io/_uploads/r1POJoR4yg.png)


# Long Streaming Inputs
Our long streaming evaluation is following [StreamingLLM](url=https://github.com/mit-han-lab/streaming-llm/).


## Usage

```
conda create -yn streaming-sepllm python=3.8
conda activate streaming-sepllm 

pip install torch torchvision torchaudio # we use torch==2.1.0+cu121 for streaming test.
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```
And to evaluate Streaming-SepLLM, you can follow this example:
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
    --output_dir ./outputs/demo/llama3_8b_sepllm_len20480_ca800_loc256_sep64_init4_pg19_demo   2>&1 | tee ./logs/demo/llama3_8b_sepllm_len20480_ca800_loc256_sep64_init4_pg19_demo.log
```
You can see other examples under ./Streaming-SepLLM/example_scripts/


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
