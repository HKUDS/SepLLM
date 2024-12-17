from ..segllm_attention import SegAttention #my
import torch

def segllm_forward_input_wrapper(forward_input, neox_args):        
    ################################################################################################ SegLLM input ##################################################################################################
    assert len(forward_input)==3, f"The length of forward_input of segllm_forward_input_wrapper must be 3, i,e, (input_ids, position_ids, attention_mask)"
    assert neox_args is not None, f"neox_args must NOT be None"

    ## Step 1: obtain a segAtten instance.
    if not neox_args.USE_ORIGINAL_FULL_ATTEN:
        segAtten = neox_args.segAtten            
        assert not (segAtten is None), "segAtten must not be None"
    else:
        pass

    ## Step 2: calculate the logical mask
    if not neox_args.USE_ORIGINAL_FULL_ATTEN:  
        attention_mask = forward_input[-1]
        input_ids = forward_input[0]
        if input_ids.shape[-1] > 1: # for prefilling
            del segAtten.past_ids[:]
            del segAtten.past_ids
            segAtten.past_ids = [] 
            segAtten.past_considered_seps_idx = [-1]
            segAtten.past_kept_tok_idx = []
            segAtten.batch_prefill_max_seq_len = input_ids.shape[-1]
            # k  = segAtten.prefill_k ## Deprecated   
            if  segAtten.PRINT_KV_RATIO: # when evaluating, print results
                segAtten.kept_tokens_count_total = ( segAtten.kept_tokens_count_seq[0] + segAtten.kept_tokens_count_total[0], segAtten.kept_tokens_count_seq[1] + segAtten.kept_tokens_count_total[1])
                segAtten.kept_attmap_count_total = ( segAtten.kept_attmap_count_seq[0] + segAtten.kept_attmap_count_total[0], segAtten.kept_attmap_count_seq[1] + segAtten.kept_attmap_count_total[1])
                segAtten.print_KV_count += 1                    

                if segAtten.print_KV_count % segAtten.print_ratio_intervals == 0:
                                            
                    print("###############################SegAttention: Kept/Total tokens for this input batch#####################################")
                    print(f" (kept, total) : {segAtten.kept_tokens_count_seq}, ratio: {(segAtten.kept_tokens_count_seq[0]+1e-6) / (segAtten.kept_tokens_count_seq[1]+1e-6)} ", flush=True)
                    print()
                    print("###############################SegAttention: Kept/Total tokens for all the inputs#####################################")
                    print(f" (kept, total) : {segAtten.kept_tokens_count_total}, ratio: { (segAtten.kept_tokens_count_total[0]+1e-6) / (segAtten.kept_tokens_count_total[1] +1e-6) } ", flush=True)
                    print()


                    print("###############################SegAttention: Kept/Total(total of lower triangular) in attention map for this input batch#####################################")
                    print(f" (kept, total) : {segAtten.kept_attmap_count_seq}, ratio: {(segAtten.kept_attmap_count_seq[0]+1e-6) / (segAtten.kept_attmap_count_seq[1]+1e-6)} ", flush=True)
                    print()
                    print("###############################SegAttention: Kept/Total(total of lower triangular) in attention map for all the inputs#####################################")
                    print(f" (kept, total) : {segAtten.kept_attmap_count_total}, ratio: { (segAtten.kept_attmap_count_total[0]+1e-6) / (segAtten.kept_attmap_count_total[1] +1e-6) } ", flush=True)
                    print()

            segAtten.kept_tokens_count_seq = (0,0)                        
            segAtten.kept_attmap_count_seq = (0,0)         
            prefill_flag = True
        else:
            # k = segAtten.generate_k  ## Deprecated
            prefill_flag = False            


        segAtten.past_ids.append(input_ids)        
        past_ids = torch.cat(segAtten.past_ids, dim=-1)
                    
        ##################### used ##########################
        # causal_mask1 = attention_mask  # B  x 1 x seqlen x seqlen ## deprecated
        # causal_mask2 = attention_mask.expand(input_ids.shape[0], attention_mask.shape[-3], attention_mask.shape[-2], attention_mask.shape[-1] )  # B x 1 x seqlen x seqlen
        causal_mask2 = attention_mask.expand(input_ids.shape[0], attention_mask.shape[-3], attention_mask.shape[-2], attention_mask.shape[-1] ).clone().detach()  # B x 1 x seqlen x seqlen
        causal_mask2 = segAtten.reverse_bool_mask(causal_mask2)
        
        init_pos_idx_tensor = None
        # del attention_mask ## pythia        

        if prefill_flag:
            if segAtten.BATCH_ADAPTIVE_INIT_POS and (not segAtten.USE_ORIGINAL_FULL_ATTEN):
                init_pos_idx_tensor, segAtten.recyc_sink_pos = segAtten.build_eval_att_sink_index(input_ids, causal_mask2, segAtten.batch_prefill_max_seq_len  ,segAtten.init_tok_max_idx + 1 , segAtten.PADDING_ID )
        else:
            if segAtten.BATCH_ADAPTIVE_INIT_POS and (not segAtten.USE_ORIGINAL_FULL_ATTEN): # when evaluating, print results. Actually no need to add this IF statement since no shift-right generation for sft
                init_pos_idx_tensor, _ = segAtten.build_eval_att_sink_index(segAtten.past_ids[0], causal_mask2, segAtten.batch_prefill_max_seq_len ,segAtten.init_tok_max_idx + 1 , segAtten.PADDING_ID, prefill_init_tok_pos_tensor=segAtten.recyc_sink_pos)

        causal_mask2 = segAtten.build_segmented_attention_mask(prefill_flag, past_ids, causal_mask2,  BATCH_ADAPTIVE_INIT_POS=segAtten.BATCH_ADAPTIVE_INIT_POS, init_pos_idx_tensor = init_pos_idx_tensor )
                    
        if  segAtten.PRINT_KV_RATIO:
            segAtten.count_prefill_kept_kv_all_layers(causal_mask2)
            segAtten.count_prefill_kept_attmap_all_layers(causal_mask2)

        if neox_args.USE_SEG_ATTN_ACCELERATOR:
            # attention_mask = segAtten.convert_to_tensor(causal_mask2)
            attention_mask = causal_mask2
        else:
            attention_mask = segAtten.reverse_bool_mask(causal_mask2, return_tensor=True)

        del causal_mask2 ##pythia 
        del init_pos_idx_tensor ##pythia 
        ##################### used ##########################
        
        forward_input = (forward_input[0], forward_input[1], attention_mask)
    else:
        pass


    ## Step 3: wrap the logical mask, kernel funcs, etc.
    if not neox_args.USE_ORIGINAL_FULL_ATTEN:
        if neox_args.USE_SEG_ATTN_ACCELERATOR:
            if neox_args.USE_BiPE:       
                input_ids = forward_input[0]
                position_ids = forward_input[1]
                attention_mask = forward_input[-1]

                intra_position_ids,  inter_position_ids = segAtten.get_bilevel_positional_ids(input_ids)
                
                seg_atten_kernel_func = segAtten.get_seg_atten_kernel_funcs(input_ids, attention_mask, neox_args.prefill_loc_win_size_list)                                        
                forward_input = (input_ids, intra_position_ids,  inter_position_ids, seg_atten_kernel_func, attention_mask)            

            else:
                seg_atten_kernel_func = segAtten.get_seg_atten_kernel_funcs(input_ids, attention_mask, neox_args.prefill_loc_win_size_list)                            
                forward_input = (input_ids, forward_input[1],  seg_atten_kernel_func, attention_mask)
                            
        else:
            if neox_args.USE_BiPE:
                input_ids = forward_input[0]
                position_ids = forward_input[1]
                attention_mask = forward_input[-1]

                intra_position_ids,  inter_position_ids = segAtten.get_bilevel_positional_ids(input_ids)
                forward_input = (input_ids, intra_position_ids,  inter_position_ids, attention_mask)
            else:
                pass
    else:
        pass
    # ################################################my Debug#################################################
    # import os
    # torch.set_printoptions(profile="full")
    # print(f">>>>>>>>>>>>>>>>>>>attention_mask for layer {3}:  RANK: {os.getenv('RANK')}:  {(~(forward_input[-1][3]))[:,:,100,:]}.<<<<<<<<<<<<<<<<<<<")
    # torch.set_printoptions(profile="default")
    # #########################################################################################################
    
    return forward_input
    ##################################################################################################################################################################################################
