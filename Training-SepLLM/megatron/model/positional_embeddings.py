# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self, dim, max_seq_len, base=10000, precision=torch.half, save_inv_freqs=False
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision
        self.max_seq_len = max_seq_len
        self.base = base
        self.dim = dim

        # precompute cos_cached, sin_cached in fp32
        cos_cached, sin_cached, inv_freq = self._prepare_cache(
            max_seq_len, precision, base
        )

        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def _prepare_cache(self, seq_len, precision, base):
        # precompute cos_cached, sin_cached in fp32
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos_cached = emb.cos()[:, None, None, :]
        sin_cached = emb.sin()[:, None, None, :]

        return (
            cos_cached.to(self.precision),
            sin_cached.to(self.precision),
            inv_freq.to(self.precision),
        )

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]

        assert seq_len <= self.max_seq_len

        if seq_len != self.max_seq_len:
            # y, z, _ = self._prepare_cache(seq_len, self.precision, self.base)
            return (
                self.cos_cached[:seq_len, ...].to(x.device),
                self.sin_cached[:seq_len, ...].to(x.device),
            )
        else:
            return self.cos_cached.to(x.device), self.sin_cached.to(x.device)


# rotary pos emb helpers:


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions

#############################################my RoPE###############################################################################
@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0, \
                        inter_position_ids=None,  ##my for BiPE
                        USE_BiPE: bool =False ##my for BiPE
                        ):
    if USE_BiPE:
        assert inter_position_ids is not None, f"When USE_BiPE={USE_BiPE}, inter_position_ids should not be {inter_position_ids}"
    '''
    ## cos.shape:[2048, 1, 1, 16], sin.shape:[2048, 1, 1, 16] #[ seq_len , batch, heads, selected_dim]; selected_dim = head_dim * neox_args.rotary_pct
    ## q.shape:[2048, 2, 12, 16], k.shape:[2048, 2, 12, 16]; #[ seq_len , batch, heads, selected_dim] ; selected_dim = head_dim * neox_args.rotary_pct
    ## inter_position_ids.shape:[2, 2048]; [bz , seq]
    '''
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...].contiguous(),
        sin[offset : q.shape[0] + offset, ...].contiguous(),
    )  ##[2048, 1, 1, 16]

    if USE_BiPE and (inter_position_ids is not None):
        cos = cos.squeeze(1).squeeze(1)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(1)  # [seq_len, dim]    

        cos = cos[inter_position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[inter_position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        cos = cos.permute(2, 0, 1, 3).contiguous() # [seq_len ,bz , 1 , dim ]
        sin = sin.permute(2, 0, 1, 3).contiguous() # [seq_len ,bz , 1 , dim ]

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(  
    q, k, cos, sin, offset: int = 0, \
        inter_position_ids=None,  ##my for BiPE
        USE_BiPE: bool =False ##my for BiPE
):  # jitting fails with bf16
    '''
    Will not be called by default Pythia.
    Have not been tested yet
    '''
    if USE_BiPE:
        assert inter_position_ids is not None, f"When USE_BiPE={USE_BiPE}, inter_position_ids should not be {inter_position_ids}"

    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )

    if USE_BiPE and (inter_position_ids is not None):
        cos = cos.squeeze(1).squeeze(1)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(1)  # [seq_len, dim]    
        
        cos = cos[inter_position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[inter_position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        cos = cos.permute(2, 0, 1, 3).contiguous() # [seq_len, bz , 1 , dim ]
        sin = sin.permute(2, 0, 1, 3).contiguous() # [seq_len, bz , 1 , dim ]

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
##############################################################################################################################




######################################################my alibi#######################################################################
class AliBi(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1, USE_BiPE=False):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ] ## shape: [Head]. 
        # print(f"Debug init slopes in __init__ slopes.shape: {slopes.shape}. slopes: {slopes} ")  ## torch.Size([12])
        # Debug init slopes in __init__ slopes.shape: torch.Size([12]). slopes: tensor([0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039, 0.7071, 0.3536, 0.1768, 0.0884]) 

        self.register_buffer("slopes", slopes)

        self.USE_BiPE = USE_BiPE

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def bias(self, seq_len_q, seq_len_k, device, dtype):
        # [b, np, sq, sk]
        # seq_len_q = x.shape[-2]
        # seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.

        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )  ## seq x seq: [2048, 2048]


            a = a.to(device).to(dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a  ## Head x seq x seq torch.Size([12, 2048, 2048]).

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return a ## Head x seq x seq, e.g., torch.Size([12, 2048, 2048]).

    def BiPE_bias(self, seq_len_q, seq_len_k, device, dtype, inter_position_ids=None):
        # [b, np, sq, sk]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.

        if self.USE_BiPE:
            assert inter_position_ids is not None, f"If USE_BiPE=True, inter_position_ids should not be None"

        target_seq_len = (
            seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
        )
        pos_ids = inter_position_ids[:, :, None].repeat(1,1,target_seq_len) # B x seq x seq

        reverse_pos_ids = -inter_position_ids[:, None, :].repeat(1, target_seq_len, 1) # B x seq x seq
        
        a = -torch.tril( pos_ids + reverse_pos_ids)[:, None, :, :] # B x 1 x seq x seq : torch.Size([32, 1, 2048, 2048])

        a = a.to(device).to(dtype) # B x 1 x seq x seq
        slopes = self.slopes.to(a.device).to(a.dtype) # [Head]
        a = a * slopes.view(1 , self.slopes.shape[0], 1, 1) ##  [B x 1 x seq x seq] *  [1, H, 1, 1] -->  [B x H x seq x seq] ： torch.Size([32, 12, 2048, 2048])

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, :,  seq_len_k - 1, :].view(
                a.shape[0], a.shape[1] , 1 , a.shape[-1]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return a


    def original_forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )  ## seq x seq: [2048, 2048]
            
            a = a.to(x.device).to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a ## Head x seq x seq torch.Size([12, 2048, 2048]).
            

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.
        
        # print(f"Debug: final a.shape: {a.shape}. a: {a} ")  ## [12, 2048, 2048]: h, seq, seq

        return x + a  


    def BiPE_forward(self, x, inter_position_ids=None):
        # [b, np, sq, sk]  : torch.Size([2, 12, 2048, 2048])  
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]        
        
        assert self.USE_BiPE, f"USE_BiPE should be True if call BiPE_forward function in AliBi class"
        assert inter_position_ids is not None, f"If USE_BiPE=True, inter_position_ids should not be None"

        a = self.BiPE_bias(seq_len_q, seq_len_k, x.device, x.dtype, inter_position_ids=inter_position_ids) ## B x H x seq x seq  : torch.Size([2, 12, 2048, 2048])        

        return x + a      

    def forward(self, x , inter_position_ids=None):
        """
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        print(f"Debug  in forward:  x.shape:{x.shape}. seq_len_q:{seq_len_q}, seq_len_k:{seq_len_k} ")
        ## Debug  in forward:  x.shape:torch.Size([2, 12, 2048, 2048]). seq_len_q:2048, seq_len_k:2048         
        """
        if self.USE_BiPE:
            assert inter_position_ids is not None, f"If USE_BiPE=True, inter_position_ids should not be None"
            return self.BiPE_forward(x, inter_position_ids)
        else:
            return self.original_forward(x)

    def get_score_bias_kernel_func(self, seq_len_q, seq_len_k, device, dtype , inter_position_ids=None):
        if self.USE_BiPE:
            assert inter_position_ids is not None, f"If USE_BiPE=True, inter_position_ids should not be None"        
            bias = self.BiPE_bias(seq_len_q, seq_len_k, device, dtype, inter_position_ids=inter_position_ids)            

            def alibi_bias_kernel(score, b, h, q_idx, kv_idx):
                return score + bias[b, h, q_idx, kv_idx]
            return alibi_bias_kernel
        else:
            bias = self.bias(seq_len_q, seq_len_k, device, dtype ) ## shape: [H x seq x seq]
            def alibi_bias_kernel(score, b, h, q_idx, kv_idx):
                return score + bias[h, q_idx, kv_idx]            
            return alibi_bias_kernel

#####################################################################################################################################
