import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from sepllm_kv_cache.kv_cache_manager import SepLLM_KVCache_Manager
from sepllm_kv_cache.utils import parse_args, load, print_args, check_args
import time
from huggingface_hub import login
# login(token="hf_xxxxxxx")  ## for your huggingface account token



args = parse_args()
check_args(args)

device = args.device
if args.dataset_name.lower() == 'pg19':
    args.dataset_name = './data/pg19/deepmind-gutenberg/deepmind-gutenberg'  
data = load_dataset(args.dataset_name, args.task, split=args.split)

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_kv_cache_manager:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif ("pythia" in model.config.model_type) or ("gpt_neox" in model.config.model_type):
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 2
    else:
        # raise ValueError(f"got {model.config.model_type}")
        raise NotImplementedError(f"NOT implemented! for the backbone type: {model.config.model_type}")
    kv_cache = SepLLM_KVCache_Manager(        
        init_cache_size=args.init_cache_size,
        sep_cache_size = args.sep_cache_size,
        local_size=args.local_size,
        cache_size = args.cache_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,                
        model_type = model.config.model_type
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from sepllm_kv_cache.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from sepllm_kv_cache.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )
        enable_falcon_pos_shift_attention(model)    
    elif ("pythia" in model.config.model_type) or ("gpt_neox" in model.config.model_type):
        from sepllm_kv_cache.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )
        enable_gpt_neox_pos_shift_attention(model)
    
    else:
        # raise ValueError(f"got {model.config.model_type}")
        raise NotImplementedError(f"NOT implemented! for the backbone type: {model.config.model_type}")
        


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

print_args(args)

num_eval_tokens = 0
total_infer_time = 0

for text in data["text"][: args.num_samples]:
    
    # print(f"text: {text}")

    encodings = tokenizer(text, return_tensors="pt")
        
    seq_len = encodings.input_ids.size(1)
    # print(f"seq_len: {seq_len}")

    pbar = tqdm(range(0, seq_len - 1))
    for idx in pbar:
    # for idx in range(0, seq_len - 1):
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        
        start_stp = time.time()
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )                        
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            if kv_cache is not None:
                kv_cache.update_past_tok_ids(input_ids)
                              
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)
            if kv_cache is not None:
                # past_key_values = kv_cache(past_key_values)
                if args.enable_SepLLM:
                    # print(f"Debug : in  if args.enable_SepLLM")
                    past_key_values = kv_cache(past_key_values, SEP_ACCUMULATION=True, USE_MAX_SEP_CACHE=True)
                elif args.enable_StreamingLLM:
                    # print(f"Debug : in  if args.enable_StreamingLLM")
                    past_key_values = kv_cache.evict_nonlocal_and_noninitial(past_key_values)
                
                else:                    
                    NotImplementedError(f"NOT implemented! for enable_kv_cache_manager={args.enable_kv_cache_manager}, and enable_SepLLM={args.enable_SepLLM}, enable_StreamingLLM={args.enable_StreamingLLM}. Please choose one between enable_SepLLM and enable_StreamingLLM if enable_kv_cache_manager={args.enable_kv_cache_manager}")
                    
        # print(f"#################num_eval_tokens:{num_eval_tokens}#####################")
        
        end_stp = time.time()
        total_infer_time += (end_stp - start_stp)

        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())

print(f"\n##############################Evaluation Summary#####################################\n")

if args.enable_kv_cache_manager:
    if args.enable_SepLLM:
        print(f"\nOverall Perplexity (PPL):  {ppl.item():.4f}\n")
        print(f"\nTotal Inference Time for generating {num_eval_tokens} tokens: {total_infer_time:.4f} seconds\n")
        print(f"\nAverage Runtime KV Usage for SepLLM: {( args.init_cache_size + args.sep_cache_size + args.local_size + args.cache_size ) / 2 }\n")
        print(f"\nMax KV Capacity: {args.cache_size }\n")
    elif args.enable_StreamingLLM:
        print(f"\nOverall Perplexity (PPL):  {ppl.item():.4f}\n")
        print(f"\nTotal Inference Time for generating {num_eval_tokens} tokens: {total_infer_time:.4f} seconds\n")    
        print(f"\nAverage Runtime KV Usage for StreamingLLM: { args.cache_size }\n")
        print(f"\nMax KV Capacity: {args.cache_size }\n")
    else:
        assert False, f"You should choose one between (enable_SepLLM, enable_SepLLM) to set it to True if enable_kv_cache_manager=True"
else:
    print(f"\nOverall Perplexity (PPL):  {ppl.item():.4f}\n")
    print(f"\nTotal Inference Time for generating {num_eval_tokens} tokens: {total_infer_time:.4f} seconds\n")
    print(f"\nAverage Runtime KV Usage: {num_eval_tokens / 2 }\n")
    print(f"\nMax KV Capacity: +infinite\n")


with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
