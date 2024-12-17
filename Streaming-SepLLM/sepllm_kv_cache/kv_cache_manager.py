import torch

def slice_on_1d(x, start, end):
    return x[:, start:end, ...]

def slice_on_2d(x, start, end):
    return x[:, :, start:end, ...]

def slice_on_3d(x, start, end):
    return x[:, :, :, start:end, ...]


def sep_1bat_select_on_1d(x, Bid , sep_index, min_sep_num=None):    
    # print(f"Debug: #####################x.shape:{x.shape} in sep_1bat_select_on_1d###########################")  ## llama: [1, 8, 97, 128], 
    sep_index = sep_index.to(x.device).detach().clone()

    if min_sep_num is None:
        return x[Bid, sep_index, ...].detach().clone()  # # Batch x seqlen x Head x dim --> sep_num x Head x dim    
    else:
        new_x =  x[Bid, sep_index, ...]  # # Batch x seqlen x Head x dim -->  sep_num x Head x dim    
        return new_x[:min_sep_num, ...].detach().clone() # #  min_sep_num x Head x dim 


def sep_1bat_select_on_2d(x, Bid, sep_index, min_sep_num=None):    
    # print(f"Debug: #####################x.shape:{x.shape} in sep_1bat_select_on_2d###########################")
    sep_index = sep_index.to(x.device).detach().clone()
    if min_sep_num is None:
        return x[Bid, :, sep_index, ...].detach().clone()  # # Batch x Head x seqlen x dim -->  Head x sep_num x dim    
    else:
        # print(f"Debug, #################x.device:{x.device}, sep_index.device:{sep_index.device}##################")
        new_x =  x[Bid, :, sep_index, ...].detach().clone()   # # Batch x Head x seqlen x dim -->  Head x sep_num x dim            
        return new_x[:, :min_sep_num, ...].detach().clone() # #  Head x min_sep_num x dim      


def sep_1bat_select_on_3d(x, Bid , sep_index, min_sep_num=None):
    # print(f"Debug: #####################x.shape:{x.shape} in sep_1bat_select_on_3d###########################")
    sep_index = sep_index.to(x.device).detach().clone()
    if min_sep_num is None:
        return x[Bid, :, :, sep_index, ...].detach().clone()  # # Batch x Head x dim x seqlen  -->  Head x dim x sep_num 
    else:
        new_x =  x[Bid, :, :,sep_index, ...].detach().clone()    # # Batch x Head x dim x seqlen  -->  Head x dim x sep_num         
        return new_x[:, :, :min_sep_num, ...].detach().clone() # #  Head x dim x min_sep_num 


DIM_TO_SLICE = {
    1: slice_on_1d,
    2: slice_on_2d,
    3: slice_on_3d,
}

BAT_DIM_TO_SELECT = {
    1: sep_1bat_select_on_1d,
    2: sep_1bat_select_on_2d,
    3: sep_1bat_select_on_3d,
}


class SepLLM_KVCache_Manager:
    def __init__(
        self,
        init_cache_size=4,        
        sep_cache_size = 32,
        local_size=224, 
        cache_size=324,        
        k_seq_dim=2,
        v_seq_dim=2,
        separator_token_ids = None,
        model_type = 'llama',
    ):
        print(f"Building SepLLM_KVCache_Manager: init_cache_size:{init_cache_size}, sep_cache_size:{sep_cache_size}, local_size:{local_size}")
        self.init_cache_size = init_cache_size
        self.local_size = local_size        
        self.cache_size = cache_size 
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        self.k_bat_dim_select = BAT_DIM_TO_SELECT[k_seq_dim]
        self.v_bat_dim_select = BAT_DIM_TO_SELECT[v_seq_dim]

        self.sep_cache_size= sep_cache_size
        
        self.past_tok_ids = None
        self.sep_exrange = 0 # runtime right boundary for separators, excluded
        self.max_sep_exidx = sep_cache_size + init_cache_size # max right boundary for separators, excluded
        
        if separator_token_ids is not None:            
            self.separator_token_ids = separator_token_ids
        else:
            if 'llama' in  model_type.lower():
                print("Debug: Here for Llama")
                self.separator_token_ids = [13, 11, 30, 0, 26, 25, 198, 220, 662, 1174, 949, 758, 2652, 551, 720, 256,262] # llama3 8b
            elif ( 'pythia' in model_type.lower() ) or ( 'gpt_neox' in model_type.lower() ):
                print("Debug: Here for GPTNeox")
                self.separator_token_ids = [15, 13, 32, 2, 28, 27, 209, 186, 187,    964, 1157, 3736, 2195, 3706, 1163, 2490,  50276,    586, 4928, 50275 ]       # pythia 14b
            elif 'falcon' in model_type.lower():
                print(f"Debug: Here for Falcon")
                self.separator_token_ids = [25, 23,  42, 12, 38, 37, 193,  4610,  204, 258, 1212, 23787, 466 ]       # falcon-40b
            else:
                raise NotImplementedError(f"NOT implemented! for the backbone type: {model_type}")
                    


    def __call__(self, past_key_values,  SEP_ACCUMULATION=True, USE_MAX_SEP_CACHE=True):
        """
        SEP_ACCUMULATION: If True, it means we will try to keep all the kv for seperators. If False, only the new_sep_kv compressed from the past_win_kv will be kept.
                                                             
        USE_MAX_SEP_CACHE: If True, it means we only keep self.sep_cache_size seperators' KV.  In the paper, the hyperparameter s is an abbreviated alias for self.sep_cache_size.


        Note: If SEP_ACCUMULATION=True and USE_MAX_SEP_CACHE=False, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and self.cache_size will be also infinitely expanded.
        """        

        return self.evict_except_for_sep(past_key_values, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE)

        

    def evict_nonlocal_and_noninitial(self, past_key_values, CHECK_LOCAL_SIZE=True):
        if CHECK_LOCAL_SIZE:
            assert self.local_size==self.cache_size-self.init_cache_size, f"For streamingLLM, cache_size==local_size+init_cache_size"

        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values        
        
        if self.init_cache_size > 0:            
            sink_kv = self.slice_kv_4_all_layers(past_key_values, 0, self.init_cache_size,  CHECK_IDX=True) 
                
        recent_kv = self.slice_kv_4_all_layers(past_key_values, seq_len - self.local_size, seq_len,  CHECK_IDX=True) 
                
        if self.init_cache_size > 0:            
            past_key_values = self.cat_kv_cache_4_all_layers([sink_kv,  recent_kv])
        else:            
            past_key_values = recent_kv
        
        return past_key_values
                
        
    def slice_kv_4_all_layers(self, past_key_values, start, end,  CHECK_IDX=False):
        if CHECK_IDX:            
            seq_len = past_key_values[0][0].size(self.k_seq_dim)
            if start <0 :
                start = start + seq_len
            if end < 0 :
                end = end + seq_len
            assert (start >=0) and (start < seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
            assert (end >= 0) and (end <= seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
            assert  start < end, f"start:{start}, end:{end}, seq_len:{seq_len}"
            
        return [
                    [ self.k_slice(k, start, end),  self.v_slice(v, start, end) ]
                    for k, v in past_key_values
              ]
        
        
    def slice_kv_cache_and_tokids(self, past_key_values, tok_ids, start, end, CHECK_IDX=True ):
        if CHECK_IDX:            
            seq_len = past_key_values[0][0].size(self.k_seq_dim)
            if start <0 :
                start = start + seq_len
            if end < 0 :
                end = end + seq_len
            assert (start >=0) and (start < seq_len), f"start:{start}, end:{end}, seq_len:{seq_len}" 
            assert (end >=0) and (end <= seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}" 
            assert  start < end, f"start:{start}, end:{end}, seq_len:{seq_len}" 
        
        
        sliced_kv = self.slice_kv_4_all_layers(past_key_values, start, end,  CHECK_IDX=False)        
        sliced_ids = tok_ids[:, start:end].detach().clone()
        
        return sliced_kv , sliced_ids

    def _cat_kv_4_all_layers(self, kv_a,  kv_b):    ## cat the KV for all layers
        assert len(kv_a) == len(kv_b)        
        return [ [ torch.cat( [kv_a[i][0], kv_b[i][0]], dim=self.k_seq_dim),  torch.cat( [kv_a[i][1], kv_b[i][1]], dim=self.v_seq_dim)    ]  for i in range(len(kv_a))  ]
                 

    def cat_kv_cache_4_all_layers(self, past_key_values_list):    
        assert len(past_key_values_list) >= 1 
        
        if len(past_key_values_list) == 1 :
            return past_key_values_list[0]
        else:
            ret = None 
            for i, past_key_values in enumerate(past_key_values_list): # enumerate all the KVs needed to be cat
                if i == 0:
                    ret = past_key_values
                else:
                    ret = self._cat_kv_4_all_layers(ret, past_key_values)
            return ret

    def cat_token_ids(self,tok_ids_list ) :
        assert len(tok_ids_list) >= 1 
        
        return torch.cat(tok_ids_list, dim=-1)        
        
    def cat_kv_cache_and_tokids(self, past_key_values_list, tok_ids_list):
        
        return self.cat_kv_cache_4_all_layers(past_key_values_list), self.cat_token_ids(tok_ids_list)
        
                                

    def compress_past_win_2_seps(self, past_win_kv, past_win_tokids, MIN_SEP_ALERT=False):

        sep_index_tensor = torch.zeros_like(past_win_tokids).bool()  # B x seq_len
        for sp_id in self.separator_token_ids:            
            sep_index_tensor = sep_index_tensor | ( past_win_tokids == sp_id ) # B x seq_len

        sep_cnt = sep_index_tensor.int().sum(-1)
        min_sep_num = sep_cnt.min()  # the min sep number for the seqs in a batch
        
        if MIN_SEP_ALERT:
            assert min_sep_num>0, f"The min sep number for each compressing time in a batch should be at least one"
                
        batch1_sep_ids_list = []
        batch_size = past_win_tokids.shape[0]
        for b_id in range(batch_size):            
            batch1_sep_ids = past_win_tokids[b_id, sep_index_tensor[b_id]] # #  sep_num
            batch1_sep_ids = batch1_sep_ids[..., :min_sep_num ].detach().clone()  # #  min_sep_num
            batch1_sep_ids_list.append(batch1_sep_ids)                                                           
            
        new_sep_tokids = torch.stack(batch1_sep_ids_list, dim=0) # #  B x min_sep_num
        
        
        new_sep_kv = []
        for k,v in past_win_kv: # for each layer
            '''
            The shape samples listed in the comments are just for Llama3.
            '''
            ## k,v torch.Size([1, 8, 129, 128])) # B x Head x seq x dim
            assert batch_size==k.shape[0]
            # batch_size = k.shape[0]
            batch1_sep_k_list = []
            batch1_sep_v_list = []
            batch1_sep_ids_list = []
            for b_id in range(batch_size):
                batch1_sep_k = self.k_bat_dim_select(k, b_id, sep_index_tensor[b_id], min_sep_num)
                batch1_sep_k_list.append(batch1_sep_k)

                batch1_sep_v = self.v_bat_dim_select(v, b_id, sep_index_tensor[b_id], min_sep_num)
                batch1_sep_v_list.append( batch1_sep_v )   
            
            sep_k = torch.stack(batch1_sep_k_list, dim=0)  ## Bx Head x min_sep_num x dim
            sep_v = torch.stack(batch1_sep_v_list, dim=0)  ## Bx Head x min_sep_num x dim           
            new_sep_kv.append( [sep_k, sep_v] )
                
        return new_sep_kv, new_sep_tokids, min_sep_num      

        
    
    def update_past_tok_ids(self, input_ids):    
        if self.past_tok_ids is None:
            self.past_tok_ids = input_ids.detach().clone()
        else:
            self.past_tok_ids = torch.cat([self.past_tok_ids , input_ids], dim=-1)
            
    def compress_kv_cache_and_tokids(self, past_key_values,  SEP_ACCUMULATION=False, USE_MAX_SEP_CACHE=False ):
        """
        SEP_ACCUMULATION: If True, it means we will try to keep all the kv for seperators. If False, only the new_sep_kv compressed from the past_win_kv will be kept.
                                                             
        USE_MAX_SEP_CACHE: If True, it means we only keep self.sep_cache_size seperators' KV.  In the paper, the hyperparameter s is an abbreviated alias for self.sep_cache_size.


        Note: If SEP_ACCUMULATION=True and USE_MAX_SEP_CACHE=False, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and self.cache_size will be also infinitely expanded.
        """        

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        
        if self.sep_exrange <=0:            
            self.sep_exrange = self.init_cache_size
                
        assert seq_len - self.local_size > self.sep_exrange
        
 
        if self.init_cache_size > 0:
            initial_kv, initial_tokids =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, 0, self.init_cache_size, CHECK_IDX=True )
                        
        Before_First_Time_Compress_Flag = (self.sep_exrange == self.init_cache_size)  ## If true, it means the present timestamp is before t1: the 1st time to compress the past window, in which only seperators' kv are kept.
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag: ## To get the old sep kv and sep token ids.           
            past_sep_kv, past_sep_tokids =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, self.init_cache_size, self.sep_exrange, CHECK_IDX=True )            
        
        past_win_kv, past_win_tokids =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, self.sep_exrange, seq_len - self.local_size, CHECK_IDX=True )        
        local_kv, local_tokids  =  self.slice_kv_cache_and_tokids( past_key_values, self.past_tok_ids, seq_len - self.local_size, seq_len, CHECK_IDX=True )
        
        new_sep_kv, new_sep_tokids, min_sep_num = self.compress_past_win_2_seps( past_win_kv, past_win_tokids) ## To get the new sep kv and sep token ids that were just compressed from the past window
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag:  ## Try to accumulate all the seen seps           
            sep_kv, sep_tokids  = self.cat_kv_cache_and_tokids( [ past_sep_kv, new_sep_kv ] ,  [past_sep_tokids, new_sep_tokids ] )                
            new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]  
        else: ## Only keep the newly obtained kv (those just compressed from the past window)
            sep_kv, sep_tokids = new_sep_kv, new_sep_tokids
            # new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]              
            assert min_sep_num==sep_len


        if USE_MAX_SEP_CACHE: ## Fixed sep cache size, i.e., only keep max_sep_len seps' kv in the cache. 
            if self.init_cache_size + sep_len > self.max_sep_exidx:
                max_sep_len = self.max_sep_exidx - self.init_cache_size
                sep_kv, sep_tokids =  self.slice_kv_cache_and_tokids( sep_kv, sep_tokids, sep_len-max_sep_len, sep_len, CHECK_IDX=True )
                self.sep_exrange =  self.max_sep_exidx  
            else:
                self.sep_exrange =  self.init_cache_size + sep_len                        
        else:    ## Extend the sep cache and the whole cache if USE_MAX_SEP_CACHE is not set                           
            self.sep_exrange =  self.init_cache_size + sep_len
            if self.sep_exrange > self.max_sep_exidx:                    
                cache_incremental_gap = self.sep_exrange - self.max_sep_exidx
                self.max_sep_exidx = self.sep_exrange 
                self.cache_size = self.cache_size + cache_incremental_gap

        if self.init_cache_size > 0:                                
            past_key_values, self.past_tok_ids  = self.cat_kv_cache_and_tokids( [initial_kv, sep_kv, local_kv ] ,  [initial_tokids, sep_tokids, local_tokids  ] )
        else:
            past_key_values, self.past_tok_ids  = self.cat_kv_cache_and_tokids( [sep_kv, local_kv ] ,  [sep_tokids, local_tokids  ] )
        
        
        return past_key_values, self.past_tok_ids
            
                        
    def evict_except_for_sep(self, past_key_values,  SEP_ACCUMULATION=True, USE_MAX_SEP_CACHE=True):
        """
        SEP_ACCUMULATION: If True, it means we will try to keep all the kv for seperators. If False, only the new_sep_kv compressed from the past_win_kv will be kept.
                                                             
        USE_MAX_SEP_CACHE: If True, it means we only keep self.sep_cache_size seperators' KV.  In the paper, the hyperparameter s is an abbreviated alias for self.sep_cache_size.


        Note: If SEP_ACCUMULATION=True and USE_MAX_SEP_CACHE=False, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and self.cache_size will be also infinitely expanded.
        """        

        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:        
            return past_key_values
        
        past_key_values, _ =   self.compress_kv_cache_and_tokids(past_key_values, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE)
        
        return past_key_values