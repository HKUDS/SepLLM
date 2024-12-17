import torch
from packaging.version import Version
###############################SegAttention Kernels#############################
from functools import lru_cache, partial

if Version(torch.__version__) >= Version('2.5.0'):
    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE,
        create_block_mask,        
    )

    # @lru_cache
    def create_seg_atten_kernel_function(seg_atten_kernel, B, H, M, N, KV_BLOCK_SIZE=128, Q_BLOCK_SIZE=128 ,   device="cuda", _compile=False):
        seg_atten_ker_func = create_block_mask(seg_atten_kernel, B, H, M, N,  BLOCK_SIZE = (KV_BLOCK_SIZE, Q_BLOCK_SIZE) ,  device=device, _compile=_compile)
        return seg_atten_ker_func
#################################################################################





#################################################my SegAttention########################################################
class SegAttention:
    def __init__(self, neox_args=None):
        self.past_considered_seps_idx = [-1]  # Store the list of the considered seps after attention sink and before non-local end. Must have "-1" as the initial placeholder. Only take effects when running random experiments
        self.past_kept_tok_idx = []  # Store the list of the random substitute kept tokens after attention sink and before non-local end. Initially empty. Only take effects when running random experiments
        self.past_ids = [] # Store the past ids. Clear it when a new question coming.
        self.kept_tokens_count_layer = [] # Store the kept tokens and total tokens of different layers.
        self.kept_tokens_count_seq =  (0,0) # Store the kept tokens and  total tokens of the final generated seq. It will be cleared every new seq
        self.kept_tokens_count_total = (0,0) # Store the kept tokens and total tokens of the total dataset. 
        
        self.kept_attmap_count_layer = [] # Store the kept tokens and total tokens of different layers.
        self.kept_attmap_count_seq =  (0,0) # Store the kept tokens and  total tokens of the final generated seq. It will be cleared every new seq
        self.kept_attmap_count_total = (0,0) # Store the kept tokens and total tokens of the total dataset. 

        self.batch_prefill_max_seq_len = -1 # Store the length of the longest seq in a batch when prefilling.
        self.FLOAT_ATT_MASK_DTYPE = torch.float32 # The availble float type for 910B3 NPU
        self.dtype = torch.float16  # for compatibility
        self.NPU_MIN = -1e37  # The min value for 910B3
        self.print_ratio_intervals = 4000
        self.print_KV_count = 0

        self.KV_BLOCK_SIZE = _DEFAULT_SPARSE_BLOCK_SIZE
        self.Q_BLOCK_SIZE  = _DEFAULT_SPARSE_BLOCK_SIZE
        
        assert self.NPU_MIN > torch.finfo(self.FLOAT_ATT_MASK_DTYPE).min

        if not (neox_args is None):
            self.Layer_num = neox_args.num_layers 
            self.RECOMPILE_SEG_ATTN_KERNEL = neox_args.RECOMPILE_SEG_ATTN_KERNEL  ## False by default. If True, recompile the Sep_Attention kernels.  When set to True, it may require more GPU memory and provide a certain level of acceleration to the training process.
        else:
            self.Layer_num = 12 
            self.RECOMPILE_SEG_ATTN_KERNEL = False ## False by default. If True, recompile the Sep_Attention kernels.  When set to True, it may require more GPU memory and provide a certain level of acceleration to the training process.
        
        ## special_tokens = ['.', ',', '?', '!', ';', ":", '\n'] 
        if neox_args is None:

            ## Pythia: GPTNeoX
            ## special_tokens = ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'] 
            self.separator_token_ids = [15, 13, 32, 2, 28, 27, 209, 186, 187] # Use [-1] to disable.  The token ids for the special tokens (i.e. separators):  ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'].
            
            self.PADDING_ID = 0  #  The id for padding token of Pythia (GPT_NeoX)
            self.prefill_k = 0   ## NOT implemented yet; From old version: Deprecated     
            self.generate_k = 0    ## NOT implemented yet; From old version: Deprecated     
            """
            The max number of layers (excluded, layers: [0, prefill_k) or [0, generate_k) ) that use the original attention masks (upper triangular matrices) when prefilling and generating respectively. These two args are NOT implemented yet and deprecated.
            For now, put a large number (>=max_seq_len) for the corresponding layers in prefill_loc_win_size_list (or generate_win_loc_size_list) if you want to keep the entire layer's KV and attentions
            """

            self.prefill_local_window_size = 10  # The local window size when training and prefilling.  KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token. Only take effect when USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS=False
            self.generate_local_window_size = 10 # The local window size when generating. KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token. Only take effect when USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS=False.        

            self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS = False  # If True: the prefilling local window sizes for different self-attention layers are different; If True: should set 'prefill_loc_win_size_list', else: should set 'prefill_local_window_size'
            self.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS = False # If True: the generating local window sizes for different self-attention layers are different;  If True: should set 'generate_win_loc_size_list', else: should set 'generate_local_window_size'
            
            self.prefill_loc_win_size_list = [2048,  256,  256,  256,  256,  2048,  256,  256,  256, 256, 256,  2048 ] # Just an example  ## The local window sizes for different self-attention layers when training (or prefilling). KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
            self.generate_win_loc_size_list = self.prefill_loc_win_size_list ## The local window sizes for different self-attention layers when generating. KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.


            self.init_tok_max_idx = 2 # The largest index for the kept initial tokens. E.g., if init_tok_max_idx==2, it means we keep 3 initial tokens (idx: 0,1,2)

            self.USE_ORIGINAL_FULL_ATTEN =False  # Flag signal with the highest priority. Run the model without any modification (standard full-attention version, i.e., standard upper triangular mask) if True.
            self.streamingLLM = False  # Run streamingLLM. Only takes effect when self.USE_ORIGINAL_FULL_ATTEN=False                        
            

            self.BATCH_ADAPTIVE_INIT_POS = False # If True: use the floating initial tokens' starting positions since when evaluating, LLM usually add paddings on the left side of the shorter seqs in a batch for alignment (i.e., left padding). Can be False when pretraining since the starting positions of initial tokens are at the beginning of each sequence in a batch for pretraining (i.e., right padding)
            self.PRINT_KV_RATIO = False # If True, print the KV cache preservation ratio (especially for the released trained model during generating). When pretraining, it will print the retention ratio for the computational complexity of calculating the attention map if it is set True
            self.print_ratio_intervals = 4000  # Print the retention ratio for the computational complexity of calculating the attention map once after every 'print_KV_intervals' forward passes (or print_KV_intervals/gradient_accumulation_steps  iterations). It only takes effect when PRINT_KV_RATIO=True.    
            self.USE_SEG_ATTN_ACCELERATOR = False  # If True, use Seg_Attention module to accelerate the training process of SegLLM
                        
            self.EXCLUDE_DIAGONAL = True # True by default and should always be True. When True, it means when we choose fixed window to process the prefilling mask, the diagonal elements in the prefilling mask could be set negative. When False: would keep the prefilling mask's diagonal positive            

            self.USE_BiPE = False   ## False by default. If True (must also set pos_emb='rotary' or 'alibi'), use Bilevel Positional Encoding.  [He, Zhenyu, et al. "Two stones hit one bird: Bilevel positional encoding for better length extrapolation." arXiv preprint arXiv:2401.16421 (2024).]
            self.BiPE_seps = [15, 13, 32, 2, 28, 27, 209, 186, 187] ## The token ids of the seperator tokens for BiPE
                        
            self.USE_SA_SOFTMAX = False # False by default. If True, use Self-Adjusting Softmax Attention.
            self.USE_SA_SOFTMAX_NO_DENO = False # False by default. If True, use Self-Adjusting Softmax Attention V2 : no denominator version.
            self.SA_Numerator_Bias = 0.0 # The bias value added to the numerator term of Self-Adjusting Softmax Attention.
            self.SA_Denominator_Bias = 1e-10 # The bias value added to the denominator term of Self-Adjusting Softmax Attention.

        else:
            ## Pythia: GPTNeoX
            ## special_tokens = ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'] 
            self.separator_token_ids = neox_args.separator_token_ids # Use [-1] to disable.  The token ids for the special tokens (i.e. separators):  ['.', ',', '?', '!', ';', ":", ' ', '\t','\n'] 

            self.PADDING_ID = neox_args.PADDING_ID   # The id for padding token of Pythia (GPT_NeoX)
            self.prefill_k = neox_args.prefill_k     ## NOT implemented yet; From old version: Deprecated   
            self.generate_k = neox_args.generate_k   ## NOT implemented yet; From old version: Deprecated  
            """
            The max number of layers (excluded, layers: [0, prefill_k) or [0, generate_k) ) that use the original attention masks (upper triangular matrices) when prefilling and generating respectively. These two args are NOT implemented yet and deprecated.
            For now, put a large number (>=max_seq_len) for the corresponding layers in prefill_loc_win_size_list (or generate_win_loc_size_list) if you want to keep the entire layer's KV and attentions
            """

            self.prefill_local_window_size = neox_args.prefill_local_window_size  # The local window size when training and prefilling.  KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token. Only take effect when USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS=False
            self.generate_local_window_size = neox_args.generate_local_window_size # The local window size when generating. KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token. Only take effect when USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS=False.        

            self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS = neox_args.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS # If True: the prefilling local window sizes for different self-attention layers are different; If True: should set 'prefill_loc_win_size_list', else: should set 'prefill_local_window_size'
            self.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS = neox_args.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS # If True: the generating local window sizes for different self-attention layers are different;  If True: should set 'generate_win_loc_size_list', else: should set 'generate_local_window_size'
            

            self.prefill_loc_win_size_list = neox_args.prefill_loc_win_size_list ## The local window sizes for different self-attention layers when training (or prefilling). KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
            self.generate_win_loc_size_list =  neox_args.generate_win_loc_size_list ## The local window sizes for different self-attention layers when generating. KVs for tokens inside the local window (we call them 'Neighboring Tokens') are kept and can been seen by the current token.
        
            self.init_tok_max_idx = neox_args.init_tok_max_idx # The largest index for the kept initial tokens. E.g., if init_tok_max_idx==2, it means we keep 3 initial tokens (idx: 0,1,2)

            self.USE_ORIGINAL_FULL_ATTEN =neox_args.USE_ORIGINAL_FULL_ATTEN  # Flag signal with the highest priority. Run the model without any modification (standard full-attention version, i.e., standard upper triangular mask) if True.
            self.streamingLLM = neox_args.streamingLLM  # Run streamingLLM. Only takes effect when self.USE_ORIGINAL_FULL_ATTEN=False                        
            

            self.BATCH_ADAPTIVE_INIT_POS = neox_args.BATCH_ADAPTIVE_INIT_POS # If True: use the floating initial tokens' starting positions since when evaluating, LLM usually add paddings on the left side of the shorter seqs in a batch for alignment (i.e., left padding). Can be False when pretraining since the starting positions of initial tokens are at the beginning of each sequence in a batch for pretraining (i.e., right padding)
            self.PRINT_KV_RATIO = neox_args.PRINT_KV_RATIO # If True, print the KV cache preservation ratio (especially for the released trained model during generating). When pretraining, it will print the retention ratio for the computational complexity of calculating the attention map if it is set True
            self.print_ratio_intervals = neox_args.print_ratio_intervals  # Print the retention ratio for the computational complexity of calculating the attention map once after every 'print_KV_intervals' forward passes (or print_KV_intervals/gradient_accumulation_steps  iterations). It only takes effect when PRINT_KV_RATIO=True.    
            self.USE_SEG_ATTN_ACCELERATOR = neox_args.USE_SEG_ATTN_ACCELERATOR # If True, use Seg_Attention module to accelerate the training process of SegLLM
                
            self.EXCLUDE_DIAGONAL = neox_args.EXCLUDE_DIAGONAL # True by default and should always be True. When True, it means when we choose fixed window to process the prefilling mask, the diagonal elements in the prefilling mask could be set negative. When False: would keep the prefilling mask's diagonal positive            

            self.USE_BiPE = neox_args.USE_BiPE   ## False by default. If True (must also set pos_emb='rotary' or 'alibi'), use Bilevel Positional Encoding.  [He, Zhenyu, et al. "Two stones hit one bird: Bilevel positional encoding for better length extrapolation." arXiv preprint arXiv:2401.16421 (2024).]
            self.BiPE_seps = neox_args.BiPE_seps  ## The token ids of the seperator tokens for BiPE
            

            self.USE_SA_SOFTMAX = neox_args.USE_SA_SOFTMAX # False by default. If True, use Self-Adjusting Softmax Attention.
            self.USE_SA_SOFTMAX_NO_DENO = neox_args.USE_SA_SOFTMAX_NO_DENO # False by default. If True, use Self-Adjusting Softmax Attention V2 : no denominator version.
            self.SA_Numerator_Bias = neox_args.SA_Numerator_Bias # The bias value added to the numerator term of Self-Adjusting Softmax Attention.
            self.SA_Denominator_Bias = neox_args.SA_Denominator_Bias # The bias value added to the denominator term of Self-Adjusting Softmax Attention.


        EXPERIMENT_NUM = int(self.USE_ORIGINAL_FULL_ATTEN) + int(self.streamingLLM) + int(self.USE_SEG_ATTN_ACCELERATOR) + int(self.USE_SA_SOFTMAX) + int(self.USE_SA_SOFTMAX_NO_DENO)
        UNIQUE_EXP_FLAG = EXPERIMENT_NUM <= 1        
        assert UNIQUE_EXP_FLAG, f"You can only set at most one True among ('USE_ORIGINAL_FULL_ATTEN'={self.USE_ORIGINAL_FULL_ATTEN}, 'streamingLLM'={self.streamingLLM} and 'USE_SEG_ATTN_ACCELERATOR'={self.USE_SEG_ATTN_ACCELERATOR}, 'USE_SA_SOFTMAX'={self.USE_SA_SOFTMAX}, 'USE_SA_SOFTMAX_NO_DENO'={self.USE_SA_SOFTMAX_NO_DENO}) at one time"

        if self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS:
            assert self.Layer_num == len(self.prefill_loc_win_size_list)
        if self.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS:
            assert  self.Layer_num == len(self.generate_win_loc_size_list)


        print("########################### Basic Args for SegAttention ############################################")
        if self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS:
            print(f"self.prefill_loc_win_size_list: {self.prefill_loc_win_size_list}")
            self.prefill_local_window_size = None
        else:        
            print(f"self.prefill_local_window_size: {self.prefill_local_window_size}")

        if self.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS:            
            print(f"self.generate_win_loc_size_list: {self.generate_win_loc_size_list}")
            self.generate_local_window_size = None
        else:
            print(f"self.generate_local_window_size: {self.generate_local_window_size}")

        print(f"self.Layer_num: {self.Layer_num}")
        print(f"self.init_tok_max_idx: {self.init_tok_max_idx}")
        print(f"self.USE_ORIGINAL_FULL_ATTEN:  {self.USE_ORIGINAL_FULL_ATTEN}" )
        print(f"self.streamingLLM:  {self.streamingLLM}" )        
        print(f"self.EXCLUDE_DIAGONAL:  {self.EXCLUDE_DIAGONAL}" )

        print(">>> Please be careful of the separator_token_ids, Make sure they are correct for the current LLM")
        print(f"self.separator_token_ids: {self.separator_token_ids}")        

        print(f"self.USE_BiPE: {self.USE_BiPE}")
        if self.USE_BiPE:
            print(f"self.BiPE_seps: {self.BiPE_seps}")
        print(f"self.USE_SA_SOFTMAX: {self.USE_SA_SOFTMAX}")
        if self.USE_SA_SOFTMAX:
            print(f"self.SA_Numerator_Bias: {self.SA_Numerator_Bias}")
            print(f"self.SA_Denominator_Bias: {self.SA_Denominator_Bias}")
        print(f"self.USE_SA_SOFTMAX_NO_DENO: {self.USE_SA_SOFTMAX_NO_DENO}")


        if int(self.USE_ORIGINAL_FULL_ATTEN)+int(self.streamingLLM) <= 0:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>--------------------- Running our SegLLM strategy ----------------------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        
        elif self.USE_ORIGINAL_FULL_ATTEN:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>--------- Running the original full attention LLM (no changing)---------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        elif self.streamingLLM:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>------------------------ Running streamingLLM --------------------------------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        else:
            print(">>>>>>>>>>>>>>>>>>>>>ERROR, ERROR, ERROR<<<<<<<<<<<<<<<<<<<<<<<<<")
            


    def count_decode_kept_element(self, mask, layer_id):
        total_toks = mask.numel()

        if  mask.dtype == torch.bool:
            kept_toks = (~(~mask)).int().sum().item()
        else:
            kept_toks = (mask>-1).int().sum().item()

        self.kept_tokens_count_layer.append( (kept_toks, total_toks) )

        if layer_id == (self.Layer_num - 1):
            self.kept_tokens_count_seq = tuple([sum(x) for x in zip(*self.kept_tokens_count_layer)])  # skip the 1st since it is for prefilling. The last element for this list is for the final seq
            self.kept_tokens_count_layer = []

    def count_prefill_kept_element(self, mask, layer_id):
        total_toks = mask.numel()

        if  mask.dtype == torch.bool:
            kept_toks = (~(~mask)).int().sum().item()
        else:
            kept_toks = (mask>-1).int().sum().item()

        self.kept_tokens_count_layer.append( (kept_toks, total_toks) )

        if layer_id == (self.Layer_num - 1):
            self.kept_tokens_count_seq = tuple([sum(x) for x in zip(*self.kept_tokens_count_layer)])  # skip the 1st since it is for prefilling. The last element for this list is for the final seq
            self.kept_tokens_count_layer = []


    def count_prefill_kept_attmap(self, mask, layer_id):
        total_entries = float(  mask.shape[0] * ( (1 + mask.shape[-1]) * mask.shape[-2] / 2 )   ) ## B * (1+ seq_len) * seq /2

        if  mask.dtype == torch.bool:
            kept_entries = (~(~mask)).int().sum().item()
        else:
            kept_entries = (mask>-1).int().sum().item()

        self.kept_attmap_count_layer.append( (kept_entries, total_entries) )

        if layer_id == (self.Layer_num - 1):
            self.kept_attmap_count_seq = tuple([sum(x) for x in zip(*self.kept_attmap_count_layer)])  # skip the 1st since it is for prefilling. The last element for this list is for the final seq
            self.kept_attmap_count_layer = []
   
    def count_prefill_kept_kv_all_layers(self, attention_mask):
        for layer_id in range(self.Layer_num):
            if isinstance(attention_mask, (list, tuple)) or len(attention_mask.shape) > 4:           
                mask = attention_mask[layer_id]
            else:
                mask = attention_mask
            
            mask_last_row = mask[:, :, -1, :]            

            self.count_prefill_kept_element(mask_last_row,  layer_id)
    
    def count_prefill_kept_attmap_all_layers(self, attention_mask):

        for layer_id in range(self.Layer_num):
            if isinstance(attention_mask, (list, tuple)) or len(attention_mask.shape) > 4:           
                mask = attention_mask[layer_id]
            else:
                mask = attention_mask
            self.count_prefill_kept_attmap(mask,  layer_id)
    


    def build_prefill_mask(self, past_ids, causal_mask2, separator_token_ids, initial_tok_size, loc_window_sizeS, BATCH_ADAPTIVE_INIT_POS=False, init_pos_idx_tensor=None, PAD_TOK_ID=-100):
        if causal_mask2.dtype == torch.bool:
            ori_causal_mask2 = causal_mask2.clone().detach()
        else:
            ori_causal_mask2 = (causal_mask2.clone().detach() > -1)

        # causal_mask2 = torch.zeros_like(ori_causal_mask2).bool().detach()  # B x 1 x seq_len x seq_len
        if isinstance(past_ids, list):
            past_ids_tensor = torch.tensor(past_ids).int().cuda()  # B x seq_len
        else:
            past_ids_tensor = past_ids.clone().detach().int().cuda()  # B x seq_len
            # past_ids_tensor = past_ids.int()  # B x seq_len ## pythia OOM. Not really useful

        ## For some code, causal_mask2'shape is B x 1 x seq_len x (seq_len+1),  last col is a pad
        if past_ids_tensor.shape[-1] != ori_causal_mask2.shape[-1]:  # Have some bugs for MMLU. I fixed it here but have not been tested.
            # print("##############paddding here##################")
            sep_index_tensor = torch.zeros( ori_causal_mask2.shape[0], ori_causal_mask2.shape[-1]  ).bool().to(past_ids_tensor.device)  # B x seq_len or B x (seq_len + 1)              
            assert not (PAD_TOK_ID in separator_token_ids),   f"PAD_TOK_ID: {PAD_TOK_ID} should not be in the separator_token_ids: {separator_token_ids}"
            assert ori_causal_mask2.shape[-1] > past_ids_tensor.shape[-1] ## For some code, causal_mask2'shape is B x 1 x seq_len x (seq_len+1  (or + n)),  last col is a pad

            pad_tensor = (torch.ones(ori_causal_mask2.shape[0], ori_causal_mask2.shape[-1] - past_ids_tensor.shape[-1] ).int() * PAD_TOK_ID).int().to(past_ids_tensor.device)     ## new version. Shape: B x 1 (or x n)
            past_ids_tensor = torch.cat([past_ids_tensor, pad_tensor], dim=-1 )  ## past_ids_tensor shoud have the same shape as sep_index_tensor. And PAD_TOK_ID should not be in the separator_token_ids
        
        else:
            sep_index_tensor = torch.zeros_like(past_ids_tensor).bool().to(past_ids_tensor.device)  # B x seq_len


        for sp_id in separator_token_ids:            
            sep_index_tensor = sep_index_tensor | ( past_ids_tensor == sp_id )# B x seq_len or B x (seq_len + 1)

        del causal_mask2 # pythia
        causal_mask2 = sep_index_tensor[:, None, None, :].expand(-1,-1,ori_causal_mask2.shape[-2],-1).clone().detach() # B x 1 x seq_len x seq_len  OR  B x 1 x seq_len x (seq_len+1)
        del sep_index_tensor # pythia

        # Initial tokens
        if BATCH_ADAPTIVE_INIT_POS  and ( not (init_pos_idx_tensor is None) ):
            # causal_mask2[init_pos_idx_tensor] = True
            assert causal_mask2.shape == init_pos_idx_tensor.shape
            assert init_pos_idx_tensor.dtype == torch.bool
            assert causal_mask2.dtype == torch.bool
            causal_mask2 = init_pos_idx_tensor | causal_mask2

        else:
            causal_mask2[:, :, : , :initial_tok_size] = True

        ## lower triangular mask. 
        # lower_tri_mask = torch.tril(torch.ones_like(causal_mask2),diagonal=0).bool() * ori_causal_mask2
        assert ori_causal_mask2.dtype == torch.bool
        lower_tri_mask =  ori_causal_mask2

        res = [] # results

        ## Local window
        loc_win_sizes_dict = {}
        if self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS:
            assert isinstance(loc_window_sizeS, (list, tuple)), "loc_window_sizeS must be a list when self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS: is True"
            for ly in range(self.Layer_num):
                if (ly > 0) and ( loc_window_sizeS[ly] == loc_window_sizeS[ly-1]  ):
                    # res.append( copy.deepcopy( res[-1] ) )
                    res.append( res[-1] )
                    continue
                elif loc_window_sizeS[ly] in loc_win_sizes_dict:
                    res.append(  loc_win_sizes_dict[ loc_window_sizeS[ly] ] )
                    continue

                w_size = - (loc_window_sizeS[ly] - int(self.EXCLUDE_DIAGONAL))
                win_mask = torch.triu(torch.ones_like(causal_mask2),diagonal=w_size).bool()                 
                # res.append(copy.deepcopy( (causal_mask2 | win_mask) & lower_tri_mask  ))
                res.append( ( (causal_mask2 | win_mask) & lower_tri_mask  ).clone().detach())
                loc_win_sizes_dict[loc_window_sizeS[ly]] = res[-1]
        else:
            window_size = loc_window_sizeS
            w_size = - (window_size - int(self.EXCLUDE_DIAGONAL))
            win_mask = torch.triu(torch.ones_like(causal_mask2),diagonal=w_size).bool()                             
            # res = copy.deepcopy( (causal_mask2 | win_mask) & lower_tri_mask  )
            res = (causal_mask2 | win_mask) & lower_tri_mask   ## pythia
            
        # ###########################################################################    
        # torch.set_printoptions(profile="full")
        # print("#####################prefill_mask res###########################")
        # print(res.shape)
        # for row in range(100):
        #     print(f"###########################prefill_mask row {row}#####################################")
        #     print(res[:, :, row, :])
        # torch.set_printoptions(profile="default") # reset
        # ###########################################################################
        return res

    def build_generate_mask(self, past_ids, causal_mask2, separator_token_ids, initial_tok_size, loc_window_sizeS, BATCH_ADAPTIVE_INIT_POS=False, init_pos_idx_tensor=None ):

        causal_mask2 = torch.zeros_like(causal_mask2).bool()  # B x 1 x 1 x seq_len
        
        if isinstance(past_ids, list):
            past_ids_tensor = torch.tensor(past_ids).int()  # B x seq_len
        else:
            past_ids_tensor = past_ids.clone().detach().int()  # B x seq_len

        sep_index_tensor = torch.zeros_like(past_ids_tensor).bool()  # B x seq_len

        for sp_id in separator_token_ids:
            # sep_index_tensor = sep_index_tensor + ( past_ids_tensor == sp_id ) # B x seq_len
            sep_index_tensor = sep_index_tensor | ( past_ids_tensor == sp_id ) # B x seq_len

        ## Set the attentions for seps to positive
        sep_index_tensor_exp = sep_index_tensor[:, None, None, :] #  B x  1 x 1 x seq_len (col)
        causal_mask2[sep_index_tensor_exp] = True

        ## Initial Tokens
        if BATCH_ADAPTIVE_INIT_POS  and ( not (init_pos_idx_tensor is None) ):
            assert causal_mask2.shape == init_pos_idx_tensor.shape
            assert init_pos_idx_tensor.dtype == torch.bool
            assert causal_mask2.dtype == torch.bool
            causal_mask2[init_pos_idx_tensor] = True
        else:
            causal_mask2[:, :, : , :initial_tok_size] = True

        res = [] # results

        ## Local window
        loc_win_sizes_dict = {}
        if self.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS:
            assert isinstance(loc_window_sizeS, (list, tuple)), "loc_window_sizeS must be a list when self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS: is True"
            for ly in range(self.Layer_num):
                if (ly > 0) and ( loc_window_sizeS[ly] == loc_window_sizeS[ly-1]  ):
                    # res.append( copy.deepcopy( res[-1] ) )
                    res.append( res[-1]  )
                    continue
                elif loc_window_sizeS[ly] in loc_win_sizes_dict:
                    res.append(  loc_win_sizes_dict[ loc_window_sizeS[ly] ] )
                    continue
                w_size = - loc_window_sizeS[ly] 
                # win_mask = torch.triu(torch.ones_like(causal_mask2),diagonal=w_size).bool() 
                win_mask = torch.zeros_like(causal_mask2).bool() ##  B x  1 x 1 x seq_len (col)
                win_mask[:,:,:,  w_size:] = True                
                res.append(( causal_mask2 | win_mask   ).clone().detach() )
                loc_win_sizes_dict[loc_window_sizeS[ly]] = res[-1]                
        else:
            window_size = loc_window_sizeS
            w_size = - window_size
            win_mask = torch.zeros_like(causal_mask2).bool() ##  B x  1 x 1 x seq_len (col)
            win_mask[:,:,:,  w_size:] = True                
            # res = copy.deepcopy( causal_mask2 + win_mask  )
            res =  causal_mask2 | win_mask 

        # ###########################################################################    
        # torch.set_printoptions(profile="full")
        # print("#####################decode_mask res###########################")
        # print(res.shape)
        # print(res[:, :, [0], :])
        # torch.set_printoptions(profile="default") # reset
        # ###########################################################################            

        return res


    def build_eval_att_sink_index(self,  input_ids, causal_mask2, pre_max_len ,initial_tok_size ,pad_id, prefill_init_tok_pos_tensor=None ):
        # assert initial_tok_size < pre_max_len, f"here f*ck, initial_tok_size:{initial_tok_size}, pre_max_len {pre_max_len} "
            
        init_pos_idx_tensor = torch.zeros_like(causal_mask2).bool()# B x  1 x seq_len (1) x seq_len
        if  not (prefill_init_tok_pos_tensor is None):
            init_tok_positions = prefill_init_tok_pos_tensor.clone().detach()
            recyc_prefill_init_tok_position_tensor = None
            assert init_tok_positions.shape[0] == init_pos_idx_tensor.shape[0], 'prefill_init_tok_pos_tensor\' s shape is wrong! Its 1st dim must be batch_size'
            assert init_tok_positions.shape[1] == initial_tok_size, 'prefill_init_tok_pos_tensor\' s shape is wrong! Its 2nd dim must be initial_tok_size'
        else:
            if isinstance(input_ids, list):
                input_ids_tensor = torch.tensor(input_ids).int()  # B x seq_len
            else:
                input_ids_tensor = input_ids.clone().detach().int()  # B x seq_len
            
            padding_num =  (input_ids_tensor[:, :pre_max_len] == pad_id).int().sum(-1) # shape: B   ; Since when decoding (generating), system will put pads at the end of the short seqs in the batch but we only need to count the pads at the left side
            init_tok_positions = padding_num[:, None].expand(-1, initial_tok_size).clone().detach() # B x initial_tok_size            
                        
            max_padding_num = padding_num.max().item()
            
            for i in range(initial_tok_size):
                init_tok_positions[:, i] += i  # B x initial_tok_size

            if max_padding_num + initial_tok_size > pre_max_len:                
                init_tok_positions = torch.clamp(init_tok_positions, max=pre_max_len-1, min=0).detach()

            recyc_prefill_init_tok_position_tensor = init_tok_positions.clone().detach()
                        # B x  1  x  1 x initial_tok_size
        
        init_tok_positions = init_tok_positions[:, None, None, :].expand(-1, init_pos_idx_tensor.shape[1], init_pos_idx_tensor.shape[2], -1  ).clone().detach() # B x  1 x seq_len (1) x initial_tok_size
        src_ones =  torch.ones_like(init_tok_positions).bool()

        init_pos_idx_tensor.scatter_(dim=-1, index=init_tok_positions, src=src_ones)

        # ###############################################################################
        # print("###################init_pos_idx_tensor#########################")        
        # torch.set_printoptions(profile="full")
        # print(init_pos_idx_tensor.shape)
        # print(init_pos_idx_tensor[:, :, [0], :])
        # torch.set_printoptions(profile="default") # reset
        # ###############################################################################

        return init_pos_idx_tensor, recyc_prefill_init_tok_position_tensor

        
    def build_segmented_attention_mask(self, prefill_flag, past_ids, causal_mask2,  BATCH_ADAPTIVE_INIT_POS=False, init_pos_idx_tensor = None ):
        if prefill_flag:                
            if self.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS:
                loc_window_sizeS = self.prefill_loc_win_size_list                
                prefill_causal_mask2_list = self.build_prefill_mask(past_ids, causal_mask2, self.separator_token_ids, self.init_tok_max_idx+1, loc_window_sizeS, BATCH_ADAPTIVE_INIT_POS=BATCH_ADAPTIVE_INIT_POS, init_pos_idx_tensor=init_pos_idx_tensor)
                return prefill_causal_mask2_list
            else:
                loc_window_sizeS = self.prefill_local_window_size                
                causal_mask2 = self.build_prefill_mask(past_ids, causal_mask2, self.separator_token_ids, self.init_tok_max_idx+1, loc_window_sizeS , BATCH_ADAPTIVE_INIT_POS=BATCH_ADAPTIVE_INIT_POS,  init_pos_idx_tensor=init_pos_idx_tensor)
                return causal_mask2              
        else:                
            if self.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS:
                loc_window_sizeS = self.generate_win_loc_size_list                
                decode_causal_mask2_list = self.build_generate_mask(past_ids, causal_mask2, self.separator_token_ids, self.init_tok_max_idx+1, loc_window_sizeS, BATCH_ADAPTIVE_INIT_POS=BATCH_ADAPTIVE_INIT_POS,  init_pos_idx_tensor=init_pos_idx_tensor )
                return decode_causal_mask2_list
            else:
                loc_window_sizeS = self.generate_local_window_size                
                causal_mask2 = self.build_generate_mask(past_ids, causal_mask2, self.separator_token_ids, self.init_tok_max_idx+1, loc_window_sizeS,  BATCH_ADAPTIVE_INIT_POS=BATCH_ADAPTIVE_INIT_POS,  init_pos_idx_tensor=init_pos_idx_tensor)   
                return causal_mask2


    def O1mask_2_infinite_mask(self, mask,  min_value, ASCEND_910B=True):        
        if mask.dtype != torch.bool:
            mask = mask.bool()

        if ('npu' in str(mask.device)) or ASCEND_910B:
            new_mask =  (~mask).float().to(dtype=self.FLOAT_ATT_MASK_DTYPE) * min_value
        else:
            new_mask = torch.zeros_like(mask).float().to(dtype=self.FLOAT_ATT_MASK_DTYPE)
            new_mask[ ~mask] = min_value
        
        return new_mask
    
    def reverse_bool_mask(self, mask, return_tensor=True):        
        if isinstance(mask, list):
            res_mask_list = []
            for i, msk in enumerate(mask):
                assert msk.dtype == torch.bool
                # mask[i] = (~msk)
                res_mask_list.append(~msk)                
            if return_tensor:
                res_mask = torch.stack(res_mask_list, dim=0)
            else:
                res_mask = res_mask_list

            del mask[:]
            del mask

            return res_mask
        else:
            assert mask.dtype == torch.bool
            return ( ~mask)

    def convert_to_tensor(self, mask):
        if isinstance(mask, (list, tuple)):
            res_mask = torch.stack(mask, dim=0)
        elif isinstance(mask, torch.Tensor):
            res_mask = mask
        return res_mask

    def get_bilevel_positional_ids(self, ids, train_scale=None):
        assert len(self.BiPE_seps)>=1, f"self.BiPE_seps:{self.BiPE_seps}. You should set self.BiPE_seps."
        sep = torch.where(ids == self.BiPE_seps[0], 1, 0)
        for i in range(len(self.BiPE_seps)):
            if i == 0:
                continue
            sep[ids == self.BiPE_seps[i]] = 1

        pos1 = torch.cumsum(sep, dim=1)  # inter-pos
        pos1 = torch.cat([torch.zeros((ids.shape[0], 1), device=ids.device), pos1[:,:-1]], dim=1)
        pos2 = torch.cat([sep, torch.ones((ids.shape[0], 1), device=ids.device)], dim=1).reshape(-1)
        ones = torch.cat([torch.zeros((1), device=ids.device) - 1, torch.argwhere(pos2 == 1)[:, 0]])
        diff = (ones[1:] - ones[:-1])
        pos2[pos2 == 1] = diff
        pos2 = -torch.cumsum(torch.cat([torch.zeros((1), device=ids.device), pos2[:-1]]), dim=0)
        pos2 = pos2 + torch.arange(pos2.shape[-1], device=ids.device)
        pos2 = pos2.reshape(ids.shape[0], -1)[:, :-1]   # intra-pos
        if train_scale is not None:
            # pos1[pos1 >= train_scale] = train_scale - 1
            pos2[pos2 >= train_scale] = train_scale - 1

        return pos2.long(), pos1.long()  ##  intra-pos,  inter-pos


    def get_seg_atten_kernel_funcs(self, input_ids, attention_mask, prefill_loc_win_size_list=None):
        B,  Sq  = input_ids.shape[0], input_ids.shape[-1], 
        Sk = Sq 
        idx0 = torch.zeros(1, dtype=torch.int, device=input_ids.device)

        if isinstance(attention_mask, (list, tuple)) or len(attention_mask.shape) > 4:
            assert prefill_loc_win_size_list is not None, f"prefill_loc_win_size_list should not be None"
            seg_atten_kernel_funcs_list = []
            seg_atten_kernel_funcs_dict = {}
            for i in range(self.Layer_num):                        
                if (i > 0) and ( prefill_loc_win_size_list[i] == prefill_loc_win_size_list[i-1]  ):                    
                    seg_atten_kernel_funcs_list.append( seg_atten_kernel_funcs_list[-1] )
                    continue
                elif prefill_loc_win_size_list[i] in seg_atten_kernel_funcs_dict:
                    seg_atten_kernel_funcs_list.append( seg_atten_kernel_funcs_dict[ prefill_loc_win_size_list[i] ] )
                    continue

                attention_mask_layer = attention_mask[i]
        
                def seg_kernel(b, h, q_idx, kv_idx):
                    aa = attention_mask_layer[b, idx0, q_idx, kv_idx]
                    return aa.view([]).detach().clone()
                
                seg_atten_kernel_func = create_seg_atten_kernel_function(seg_kernel, B, 1, Sq, Sk, KV_BLOCK_SIZE=self.KV_BLOCK_SIZE, Q_BLOCK_SIZE=self.Q_BLOCK_SIZE, device=input_ids.device, _compile=self.RECOMPILE_SEG_ATTN_KERNEL)
                seg_atten_kernel_funcs_list.append(seg_atten_kernel_func)
                seg_atten_kernel_funcs_dict[ prefill_loc_win_size_list[i] ] = seg_atten_kernel_func

            return seg_atten_kernel_funcs_list

        else:            
            def seg_kernel(b, h, q_idx, kv_idx):
                aa = attention_mask[b, idx0, q_idx, kv_idx]                
                return aa.view([]).detach().clone()

            seg_atten_kernel_func = create_seg_atten_kernel_function(seg_kernel, B, 1, Sq, Sk, KV_BLOCK_SIZE=self.KV_BLOCK_SIZE, Q_BLOCK_SIZE=self.Q_BLOCK_SIZE, device=input_ids.device, _compile=self.RECOMPILE_SEG_ATTN_KERNEL)
            return seg_atten_kernel_func

###########################################################################################################
