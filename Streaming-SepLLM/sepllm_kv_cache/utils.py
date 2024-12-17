import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import os.path as osp
import ssl
import urllib.request
import os
import json

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="The name or path (Local or HuggingFace) for the backbone models")
    parser.add_argument("--output_dir", type=str, default="outputs/debug", help="The directory to be used for saving the evaluation results" )


    parser.add_argument("--dataset_name", type=str, default="wikitext",  choices=["wikitext", "pg19"],  help="The data set to use to evaluate our methods" ) 
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1", choices=["wikitext-2-raw-v1", "default"] ,  help="The sub task of the data set to use to evaluate our methods" )   
    
    # parser.add_argument("--dataset_name", type=str, default="PG19")  ##my
    # parser.add_argument("--task", type=str, default='default')   ## my
    
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"], help="The split of the dataset to use to evaluate our methods")
    parser.add_argument("--num_samples", type=int, default=3000000, help="The max number of samples to be used for evaluation" )
    parser.add_argument("--num_eval_tokens", type=int, default=20*1024, help="The number of tokens the model needs to generate during testing; in actual execution, the number of tokens generated is the smaller value between the tokens corresponding to num_samples and num_eval_tokens")
    
    parser.add_argument("--enable_kv_cache_manager", type=str2bool, default=True, help="If True, the KV Cache Manager is enabled to manage our KV cache; otherwise, it is not enabled.")
    parser.add_argument("--enable_SepLLM", type=str2bool , default=True, help="If True, SepLLM is enabled to manage our KV cache for long streaming inputs; otherwise, it is not enabled. Must keep enable_kv_cache_manager=True and enable_StreamingLLM=False if you want to use enable_SepLLM=True")
    parser.add_argument("--enable_StreamingLLM", type=str2bool , default=False, help="If True, StreamingLLM is enabled to manage our KV cache; otherwise, it is not enabled. Must keep enable_kv_cache_manager=True and enable_SepLLM=False if you want to use enable_StreamingLLM=True")

    parser.add_argument("--cache_size", type=int, default=324, help="Only take effect when enable_kv_cache_manager=True. The max capacity of the whole KV cache, i.e., hyperparameter c in the paper")
    parser.add_argument("--init_cache_size", type=int, default=4, help="Only take effect when enable_SepLLM=True or enable_StreamingLLM=True. init_cache_size is the max number of KVs for the initial tokens that are kept in cache.")
    parser.add_argument("--sep_cache_size", type=int, default=64, help="Only take effect when enable_SepLLM=True. It means the max capacity of Separator Cache, i.e., hyperparameter s in the paper.")
    parser.add_argument("--local_size", type=int, default=224, help="Only take effect when enable_SepLLM=True or enable_StreamingLLM=True. For enable_SepLLM=True, it is the hyperparameter w, i.e., The min number of KV of contiguous Neighboring Tokens that should be kept in the KV cache after the entire KV cache is initially fully filled (triggering the first compression operation). For enable_StreamingLLM=True, it means the number of KV of local tokens that should be kept in the KV cache")    
    parser.add_argument("--enable_pos_shift", type=str2bool, default=True, help="If True, it will enable the Positional Encoding Shifting, i.e., we focus on positions within the cache instead of those in the original text")


    parser.add_argument("--device", type=str, default="cuda:0", help="The device to use")

    args = parser.parse_args()
    return args

def check_args(args):
    if args.enable_kv_cache_manager:
        assert args.cache_size > 0 , f"cache_size must be greater than 0"
        assert args.init_cache_size >= 0 , f"init_cache_size must be greater than (equal to) 0"
        assert args.local_size > 0 , f"local_size must be greater than 0"
        assert args.cache_size >= args.init_cache_size + args.local_size , f"cache_size must be greater than (or equal to) init_cache_size + local_size"

        assert int(args.enable_SepLLM) + int(args.enable_StreamingLLM) > 0, f"If enable_kv_cache_manager=True, you must choose one between enable_SepLLM and enable_StreamingLLM and set it to True"
        assert int(args.enable_SepLLM) + int(args.enable_StreamingLLM) < 2, f"If enable_kv_cache_manager=True, you must choose JUST ONE between enable_SepLLM and enable_StreamingLLM to set it to True"
        if args.enable_StreamingLLM:
            print(f"Warnings: if enable_StreamingLLM=True. sep_cache_size={args.sep_cache_size} will NOT take effect")
            assert args.local_size==args.cache_size-args.init_cache_size, f"For streamingLLM, cache_size==local_size+init_cache_size"
        if args.enable_SepLLM:
            assert args.sep_cache_size > 0 , f"sep_cache_size must be greater than 0 if enable_SepLLM=True"
            assert args.init_cache_size + args.sep_cache_size + args.local_size < args.cache_size, f"init_cache_size({args.init_cache_size}) + sep_cache_size({args.sep_cache_size}) + local_size:({args.local_size}) should be less than cache_size:({args.cache_size}), i.e., a + s + w < c"
    else:
        if args.enable_StreamingLLM:
            print(f"Warnings: if enable_kv_cache_manager=False. enable_StreamingLLM={args.enable_StreamingLLM} will NOT take effect")
        if args.enable_SepLLM:
            print(f"Warnings: if enable_kv_cache_manager=False. enable_SepLLM={args.enable_SepLLM} will NOT take effect")


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict
