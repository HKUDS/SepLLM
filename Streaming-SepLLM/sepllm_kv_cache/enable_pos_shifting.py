from sepllm_kv_cache.kv_cache_manager import SepLLM_KVCache_Manager


def enable_pos_shifting(model, args):
    if "llama" in model.config.model_type.lower():
        k_seq_dim = 2
        v_seq_dim = 2
        from sepllm_kv_cache.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)

    elif ("gpt_neox" in model.config.model_type.lower()) or ("pythia" in model.config.model_type.lower()):
        k_seq_dim = 2
        v_seq_dim = 2
        from sepllm_kv_cache.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type.lower():
        v_seq_dim = 2
        k_seq_dim = 2
        from sepllm_kv_cache.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
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

    return kv_cache
