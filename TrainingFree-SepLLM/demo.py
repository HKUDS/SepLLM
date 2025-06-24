from transformers import AutoTokenizer, AutoModelForCausalLM, SepCache
import torch
from huggingface_hub import login
login("hf_xxxXXXxxx")


def to_cuda(a_dict: dict) -> dict:
    new_dict = {}    
    for k,v in a_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.cuda()
        else:
            new_dict[k] = v
    return new_dict

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", attn_implementation="flash_attention_2", device_map="cuda:0")
model.bfloat16().cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
inputs = tokenizer(text=["My name is Llama 3"], return_tensors="pt")
inputs = to_cuda(inputs)

"""
# For initializing `SepCache`:
# `separator_token_ids: List[int] ` is required for SepCache's initialization if `model_type` is not provided.
# `PADDING_ID: int` is required for SepCache's initialization if `model_type` is not provided.
"""
past_key_values = SepCache(init_cache_size=4,sep_cache_size=128,local_size=256,cache_size=512, layer_num=32, USE_MAX_SEP_CACHE=True, model_type='llama')
outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
outputs.past_key_values # access cache filled with keys/values from generation


## When using the `update` function of SepCache to update the keys/values and the past token ids (necessary in SepCache), the current `input_ids` must also be provided. 
# key_states, value_states = past_key_values.update(                
#         key_states = key_states,
#         value_states = value_states,    
#         input_ids = input_ids,
#         layer_idx = layer_idx,     
#         PREFILLING_FLAG = q_len > 1, ## `q_len` is the sequence length of the current `query_states`
#         )