# Copyright (c) 2024 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from pkg_resources import packaging
from importlib.metadata import version

from .norms import get_norm
from megatron import mpu
from megatron.model import megablocks_utils
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.activations import get_activation
from megatron.model.utils import exists, get_fusion_type
from megatron.model.positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb,
    AliBi,
)
from megatron.model.fused_rope import (
    FusedRoPEFunc,
    fused_apply_rotary_pos_emb_cached,
)
from megatron.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from megatron.model.utils import configure_sparse_attention
from deepspeed.moe.layer import MoE

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     kv: number of key or value heads
     p: number of model parallel partitions
     np: n/p
     kvp: kv/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""



###############################SepAttention Kernels#############################
from packaging.version import Version

if Version(torch.__version__) >= Version('2.5.0'):
    from torch.nn.attention.flex_attention import (                        
        flex_attention,
    )
    # torch._dynamo.config.suppress_errors = True
    flex_attention = torch.compile(flex_attention, dynamic=False)

def remove_none(args):
    return tuple(el for el in args if el is not None)
#################################################################################






class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        MOE=False,
        MoE_mp_size=1,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = int(4 * 2 / 3) if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * neox_args.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim_in,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            parallel_output=parallel_output,
            skip_bias_add=True,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Note: multiple_of is used to compute the hidden dimension of the MLP
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
        MOE=False,
        MoE_mp_size=1,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        # Allow custom intermediate size, e.g. for Mistral
        if neox_args.intermediate_size is not None:
            ff_dim = neox_args.intermediate_size
        else:
            ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
            ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w3 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        neox_args,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if neox_args.use_mup = True, despite it not being included here
                seq_dim=1,  # important: must mark that this layer receives shape [b, s, h] not [s, b, h] and so Seq. Parallel comms must gather along dim=1 rather than dim=0
            )

    #        else:
    #            print(
    #                'ERROR: Output layer parallelism over the hidden dim is currently broken (https://github.com/EleutherAI/gpt-neox/issues/905). Please run with output_layer_parallelism = "column" until this issue is fixed.'
    #            )
    #            exit()
    #            self.final_linear = mpu.RowParallelLinear(
    #                neox_args=neox_args,
    #                input_size=neox_args.hidden_size,
    #                output_size=neox_args.padded_vocab_size,
    #                bias=False,
    #                input_is_parallel=False,
    #                init_method=init_method,
    #                parallel_output=parallel_output,
    #                skip_bias_add=False,
    #                mup_rescale_parameters=is_last_layer,  # only called if neox_args.use_mup = True, despite it not being included here
    #            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class _MegablocksAdapter(nn.Module):
    def __init__(
        self, neox_args, layer_cls, init_method, output_layer_init_method, ep_group
    ):
        super().__init__()
        megablocks_utils.assert_megablocks_is_available()
        args = megablocks_utils.as_megablocks_args(neox_args)
        args.device = torch.cuda.current_device()
        args.init_method = init_method
        args.output_layer_init_method = output_layer_init_method

        # NOTE: Shard the MoE layers over the data parallel group. Expert
        # parallel sharding and data parallel sharding could be decoupled
        # by extending the optimizer to handle data parallel reductions for
        # MoE and non-MoE parameters separately.
        if args.moe_expert_model_parallelism:
            args.expert_parallel_group = ep_group

        if neox_args.moe_glu:
            args.mlp_type = "glu"

        self.moe = layer_cls(args)

    def forward(self, x):
        return self.moe.forward(x)


class MbMoE(_MegablocksAdapter):
    def __init__(self, neox_args, init_method, output_layer_init_method, ep_group):
        super().__init__(
            neox_args,
            megablocks_utils.moe.MoE,
            init_method,
            output_layer_init_method,
            ep_group,
        )


class dMoE(_MegablocksAdapter):
    def __init__(self, neox_args, init_method, output_layer_init_method, ep_group):
        super().__init__(
            neox_args,
            megablocks_utils.dmoe.dMoE,
            init_method,
            output_layer_init_method,
            ep_group,
        )


class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb
        ##########################SepLLM################################
        self.USE_BiPE  = neox_args.USE_BiPE  ##my
        self.USE_SA_SOFTMAX = neox_args.USE_SA_SOFTMAX ##my
        self.USE_SEP_ATTN_KERNEL_ACCELERATOR = neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR        
        self.USE_SA_SOFTMAX_NO_DENO = neox_args.USE_SA_SOFTMAX_NO_DENO
        self.USE_ORIGINAL_FULL_ATTEN = neox_args.USE_ORIGINAL_FULL_ATTEN

        assert rotary == (self.pos_emb =='rotary' )  ##my
        ##########################################################
        

        self.use_qk_layernorm = neox_args.use_qk_layernorm
        if self.use_qk_layernorm:
            norm, eps = get_norm(neox_args)
            self.qk_layernorm = norm(
                [
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                ],
                eps=eps,
            )

        self.sliding_window_width = neox_args.sliding_window_width

        if (
            not neox_args.num_kv_heads
            or neox_args.num_kv_heads == neox_args.num_attention_heads
        ):
            self.gqa = False
        else:
            self.gqa = True
        if self.gqa:
            self.num_kv_heads_per_partition = mpu.divide(
                neox_args.num_kv_heads, world_size
            )  # we do not yet clone KV heads in MQA across TP ranks...
            self.kv_hidden_size = (
                neox_args.num_kv_heads * self.hidden_size_per_attention_head
            )  # how large the total hidden dim for each of K and V is
        else:
            self.num_kv_heads_per_partition = self.num_attention_heads_per_partition
            self.kv_hidden_size = neox_args.hidden_size

        if not self.gqa:
            # Strided linear layer.
            self.query_key_value = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=3 * neox_args.hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=neox_args.use_bias_in_attn_linear,
            )
        else:
            # QKV proj is smaller if we are using GQA / MQA
            self.query_key_value = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.hidden_size + 2 * self.kv_hidden_size,
                gather_output=False,
                init_method=init_method,
                bias=neox_args.use_bias_in_attn_linear,
            )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
                USE_BiPE = neox_args.USE_BiPE
            )

        # TODO: this arg shouldn't need to be passed in - get from neox_args
        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct  ##my: rotary_pct=0.25 for pythia-160m
                )  ###my: self.rotary_ndims:16 for pythia-160M
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )  ##my, dim = 16 for for pythia-160M
            # print(f"Debug in ParallelSelfAttention: dim:{dim}")

            self.rotary_emb = RotaryEmbedding(
                dim,
                base=neox_args.rotary_emb_base,
                max_seq_len=neox_args.seq_length,
                precision=neox_args.params_dtype,
                save_inv_freqs=neox_args.rotary_save_freqs_buffer,
            )
        else:
            self.rotary_emb = None

        self.rope_fusion = neox_args.rope_fusion ##my. For default pythia: False
        self.attention_type = neox_args.attention_config[layer_number]        
        self.sparse = self.attention_type != "global"

        if self.gqa:
            assert not self.sparse

        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:

            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=self.fp16,
                input_in_bf16=self.bf16,
                fusion_type=get_fusion_type(neox_args),
                mask_func=self.attention_mask_func,
                softmax_in_fp32=self.attention_softmax_in_fp32,
                scale=coeff,
            )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.dropout_p = neox_args.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=neox_args.use_bias_in_attn_linear,
        )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask, sep_atten_kernel_func=None , inter_position_ids=None
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]                        
        if self.USE_SEP_ATTN_KERNEL_ACCELERATOR and ( not self.USE_ORIGINAL_FULL_ATTEN ):
            query = query_layer.permute(1,2,0,3).contiguous() ## B x head x seq x d: torch.Size([32, 12, 2048, 64])
            key = key_layer.permute(1,2,0,3).contiguous()  ## B x head x seq x d:  torch.Size([32, 12, 2048, 64])
            value = value_layer.permute(1,2,0,3).contiguous() ## B x head x seq x d: torch.Size([32, 12, 2048, 64])

            # query = query_layer.permute(1,2,0,3)  ## B x head x seq x d: torch.Size([32, 12, 2048, 64])
            # key = key_layer.permute(1,2,0,3)  ## B x head x seq x d:  torch.Size([32, 12, 2048, 64])
            # value = value_layer.permute(1,2,0,3)  ## B x head x seq x d: torch.Size([32, 12, 2048, 64])

            seq_len_q = query.shape[-2]
            seq_len_k = key.shape[-2]
            
            if self.pos_emb == "alibi" :                
                pos_bias_ker_func = self.alibi_embed.get_score_bias_kernel_func( seq_len_q, seq_len_k, value.device, value.dtype , inter_position_ids=inter_position_ids)                                
            else:
                pos_bias_ker_func = None

            assert sep_atten_kernel_func is not None, f"If USE_SEP_ATTN_KERNEL_ACCELERATOR={self.USE_SEP_ATTN_KERNEL_ACCELERATOR}, sep_atten_kernel_func should NOT be None"
                                    
            context_layer =  flex_attention(query, key, value, score_mod=pos_bias_ker_func, block_mask=sep_atten_kernel_func)        
            return context_layer
        else:

            output_size = (
                query_layer.size(1),
                query_layer.size(2),
                query_layer.size(0),
                key_layer.size(0),
            )
            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(
                output_size[2], output_size[0] * output_size[1], -1
            )
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
            # preallocating result tensor: [b * np, sq, sk]
            matmul_result = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=torch.cuda.current_device(),
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)
            # ==================================================
            # Update attention mask for inference. [b, np, sq, sk]
            # ==================================================

            if self.use_cache:
                with torch.no_grad():
                    attention_mask = attention_mask[
                        ..., : attention_scores.size(3), : attention_scores.size(3)
                    ]

            # ===========================
            # Attention probs and dropout
            # ===========================

            if exists(self.rpe):
                rpe = self.rpe(query_layer.size(0), key_layer.size(0))
                attention_scores += rpe  # [1, np, sq, sk]

            if self.pos_emb == "alibi":
                attention_scores = self.alibi_embed(attention_scores, inter_position_ids=inter_position_ids)

            ############################################my: For Self-Adjusting Softmax Attention###########################################################
            if self.USE_SA_SOFTMAX and ( not self.USE_ORIGINAL_FULL_ATTEN ):
                attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

                attention_scores_triled = torch.tril(attention_scores)
        
                min_attention_score=torch.min(attention_scores_triled, -1, keepdim=True)[0]
                max_attention_score=torch.max(attention_scores_triled, -1, keepdim=True)[0] + self.SA_Denominator_Bias
                
                min_attention_score=torch.minimum(min_attention_score, torch.zeros(1, device= attention_scores.device))
                max_attention_score=torch.maximum(max_attention_score, torch.zeros(1, device= attention_scores.device)) + self.SA_Denominator_Bias
                                
                attention_probs=attention_probs * ( (attention_scores - min_attention_score + self.SA_Numerator_Bias)  / (max_attention_score - min_attention_score ) )                            
                attention_probs = torch.tril(attention_probs).half()
                            
            elif self.USE_SA_SOFTMAX_NO_DENO and ( not self.USE_ORIGINAL_FULL_ATTEN ):                    
                    attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
                    attention_probs=attention_probs * attention_scores
                    attention_probs=torch.tril(attention_probs)

            else: ##my: default
                # attention scores and attention mask [b, np, sq, sk]
                attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)    

            ############################################my: For Self-Adjusting Softmax Attention###########################################################


            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)

            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (
                value_layer.size(1),
                value_layer.size(2),
                query_layer.size(0),
                value_layer.size(3),
            )

            # change view [sk, b * np, hn]
            value_layer = value_layer.view(
                value_layer.size(0), output_size[0] * output_size[1], -1
            )

            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(
                output_size[0] * output_size[1], output_size[2], -1
            )

            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            return context_layer               
  

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

    def gqa_project(self, hidden_states, attention_mask, layer_past=None):
        # QKV projection and separation into separate Q/K/V layers for GQA,
        # where KV projections may be smaller than Q projection.
        # the logic for this is explained in comments of this function
        # detailing the intermediate sizes of tensors at each reshape.

        # pass through projection: [sq, b, h] --> [sq, b, ((np + 2 * kvp) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # First: reshape so we have seqlen, batch, and num. query heads each as separate dims
        # Final dim is not exactly head dim: the first (head dim) dims are query heads,
        # The last (head dim * ratio of kv to q heads) each are the "k/v heads"
        # (right now we treat like we have same num. heads, but smaller head dim)

        # [sq, b, ((np + 2 * kvp) * hn)] --> [sq, b, np, (hn * (1 + 2 * (kvp / np)))]
        new_qkv_shape = (
            mixed_x_layer.shape[0],
            mixed_x_layer.shape[1],
            self.num_attention_heads_per_partition,
            int(
                self.hidden_size_per_attention_head
                * (
                    1
                    + 2
                    * (
                        self.num_kv_heads_per_partition
                        / self.num_attention_heads_per_partition
                    )
                )
            ),
        )
        mixed_x_layer = mixed_x_layer.reshape(*new_qkv_shape)

        # Next: split our fake head dim. (last dim) so that the first (head dim) dimensions go to Q,
        # the last smaller 2 * (head dim * kv to q head ratio) each divided between K and V separately
        split_sizes = (
            self.hidden_size_per_attention_head,
            int(
                (
                    self.num_kv_heads_per_partition
                    / self.num_attention_heads_per_partition
                )
                * self.hidden_size_per_attention_head
            ),
            int(
                (
                    self.num_kv_heads_per_partition
                    / self.num_attention_heads_per_partition
                )
                * self.hidden_size_per_attention_head
            ),
        )

        # [sq, b, np, (hn * (1 + 2 * (kvp / np)))] --> 1 x [sq, b, np, hn] , 2 x [sq, b, np, (hn * (kvp / np))]
        (query_layer, key_layer, value_layer) = [
            x.contiguous()
            for x in torch.split(
                mixed_x_layer,
                split_sizes,
                dim=mixed_x_layer.dim() - 1,
            )
        ]

        # reshape K/V to proper output shape (last dim = correct full "real" head size again)
        # 2 x [sq, b, np, (hn * (kvp / np))] --> 2 x [sq, b, kvp, hn]
        new_kv_shape = (
            key_layer.size(0),
            key_layer.size(1),
            self.num_kv_heads_per_partition,
            self.hidden_size_per_attention_head,
        )

        key_layer = key_layer.view(*new_kv_shape)

        value_layer = value_layer.view(*new_kv_shape)

        # We repeat K/V heads to match Q head counts    
        key_layer = torch.repeat_interleave(
            key_layer,
            repeats=int(
                self.num_attention_heads_per_partition
                // self.num_kv_heads_per_partition
            ),
            dim=2,
        )
        value_layer = torch.repeat_interleave(
            value_layer,
            repeats=int(
                self.num_attention_heads_per_partition
                // self.num_kv_heads_per_partition
            ),
            dim=2,
        )

        return query_layer, key_layer, value_layer

    def forward(self, hidden_states, attention_mask, layer_past=None, sep_atten_kernel_func=None, inter_position_ids=None):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        if not self.gqa:
            # QKV projection for MHA.

            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
        else:
            # Grouped Query Attention (GQA) - specific logic for performing QKV proj
            # and separating out Q, K, and V outputs.

            # output shapes: 1 x [sq, b, np, hn]
            query_layer, key_layer, value_layer = self.gqa_project(
                hidden_states, attention_mask, layer_past=layer_past
            )

        # QK Normalization https://arxiv.org/abs/2302.05442
        if self.use_qk_layernorm:
            query_layer = self.qk_layernorm(query_layer)
            key_layer = self.qk_layernorm(key_layer)

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims): 
                
                # print(f"Debug: in partial rotary: exists(self.rotary_ndims)=True: self.rotary_ndims:{self.rotary_ndims} ")
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer

            seq_len = key_layer.shape[0]
            offset = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            ###################################SepLLM#####################################
            #[ seq_len , batch, heads, dim]
            # print(f"Debug: query_layer.shape:{query_layer.shape}, key_layer.shape:{key_layer.shape} ") 
            # #Debug: query_layer.shape:torch.Size([2048, 2, 12, 64]), key_layer.shape:torch.Size([2048, 2, 12, 64])
            # print(f"Debug: query_rot.shape:{query_rot.shape}, key_rot.shape:{key_rot.shape} ")
            # # Debug: query_rot.shape:torch.Size([2048, 2, 12, 16]), key_rot.shape:torch.Size([2048, 2, 12, 16])

            # if self.USE_BiPE:
            #     cos, sin = self.rotary_emb(value_layer, seq_len=seq_len )
            # else:
            #     cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)

            
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len ) ##Keep default
            ##########################################################################

            if self.rope_fusion: ##my. For default pythia: False
                query_layer, key_layer = (
                    fused_apply_rotary_pos_emb_cached(rot, cos, sin)
                    for rot in [query_rot, key_rot]
                )
            else:
                if self.bf16: ##my. Default Pythia: self.bf16=False
                    apply_rotary_fn = apply_rotary_pos_emb_torch
                    # print(f"Debug: here: self.bf16: {self.bf16} ")
                else:
                    # print(f"Debug: here: self.bf16: {self.bf16} ")
                    apply_rotary_fn = apply_rotary_pos_emb

                ###########################################my BiPE############################################
                query_layer, key_layer = apply_rotary_fn(  ##my 
                    query_rot, key_rot, cos, sin, offset=offset, inter_position_ids=inter_position_ids, USE_BiPE=self.USE_BiPE
                )
                ##############################################################################################

            if exists(self.rotary_ndims):  ##my :for Pythia-160: rotary_ndims=16
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        if self.sparse:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )
        else:
            context_layer = self.attention(
                query_layer, key_layer, value_layer, layer_past, attention_mask, sep_atten_kernel_func=sep_atten_kernel_func, inter_position_ids=inter_position_ids
            )


        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, present]

        return output, bias


class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):

        super().__init__()
        self.layer_number = layer_number
        self.neox_args = neox_args

        norm, eps = get_norm(neox_args)

        # Layernorm on the input data.
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.use_cache = use_cache

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.gpt_j_residual = neox_args.gpt_j_residual
        self.gpt_j_tied = neox_args.gpt_j_tied
        self.mlp_type = neox_args.mlp_type
        self.moe_type = neox_args.moe_type

        ################################my######################################
        self.USE_BiPE = neox_args.USE_BiPE ##my. For BiPE
        self.USE_SEP_ATTN_KERNEL_ACCELERATOR = neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR ##my. For Flex attention
        ######################################################################

        if self.gpt_j_residual:
            # GPT-J style layers allow us to defer the reduction of results across TP ranks until the end of the two sublayers.
            # the reduction we use is a simple allreduce for pure Tensor Parallel,
            # but needs to be a reduce-scatter when using Megatron-style Sequence Parallel (LN sharding.)
            self.reduce = (
                mpu.mappings.reduce_from_model_parallel_region
                if not neox_args.sequence_parallel
                else mpu.mappings.reduce_scatter_to_sequence_parallel_region
            )

        # Self attention.
        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            use_cache=self.use_cache,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )

        # Layernorm on the output of the attention layer.
        # If GPT-J residuals are used, this is surpurfulous but leaving it in
        # leads to cleaner code
        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)

        # MLP
        def get_mlp(mlp_type, **kw):
            if mlp_type == "regular":
                return ParallelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                    **kw,
                )
            elif mlp_type == "llama":
                return LLaMAParallelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    parallel_output=self.gpt_j_residual,
                    **kw,
                )
            else:
                raise KeyError(mlp_type)

        self.num_experts = (
            neox_args.moe_num_experts
            if layer_number % neox_args.expert_interval == 0
            else 1
        )
        args = neox_args
        if self.num_experts <= 1:
            self.mlp = get_mlp(neox_args.mlp_type)
        else:
            from torch import distributed as dist

            if self.num_experts > dist.get_world_size():
                moe_mp_size = 1
            else:
                moe_mp_size = dist.get_world_size() // self.num_experts

            if neox_args.moe_type == "deepspeed":
                self.mlp = MoE(
                    args.hidden_size,
                    get_mlp(
                        "regular",
                        MOE=True,
                        MoE_mp_size=moe_mp_size,
                    ),
                    num_experts=self.num_experts,
                    ep_size=args.moe_expert_parallel_size,
                    k=args.moe_top_k,
                    use_residual=args.moe_use_residual,
                    capacity_factor=args.moe_train_capacity_factor,
                    eval_capacity_factor=args.moe_eval_capacity_factor,
                    min_capacity=args.moe_min_capacity,
                    drop_tokens=args.moe_token_dropping,
                    use_tutel=args.use_tutel,
                    enable_expert_tensor_parallelism=args.enable_expert_tensor_parallelism,
                )
            elif neox_args.moe_type == "megablocks":

                def integrate_megablocks_with_ds_expert_parallelism():
                    # We make megablocks work with DS parallelism.
                    #
                    # We fool DS into accepting these MoE parameters as its own DS MoE params,
                    # which makes things work with the underlying expert parallelism,
                    # including TED parallelism.
                    #
                    # Effectively, we want to:
                    #
                    # - Make DS's data parallel gradient all-reduction skip these params.
                    # - But make these params participate in the expert parallel all-reduction!
                    #
                    # Further background:
                    #
                    # Normally, with the original megablocks demo codebase, it
                    # only supports 1 copy of any expert throughout
                    # the network, since it uses EP group = DP group.
                    #
                    # First, we trigger DS initialization of the MoE expert parallel groups and internal state.
                    throwaway = MoE(
                        args.hidden_size,
                        get_mlp(
                            "regular",
                            MOE=True,
                            MoE_mp_size=moe_mp_size,
                        ),
                        num_experts=self.num_experts,
                        ep_size=args.moe_expert_parallel_size,
                        k=args.moe_top_k,
                        use_residual=args.moe_use_residual,
                        capacity_factor=args.moe_train_capacity_factor,
                        eval_capacity_factor=args.moe_eval_capacity_factor,
                        min_capacity=args.moe_min_capacity,
                        drop_tokens=args.moe_token_dropping,
                        use_tutel=args.use_tutel,
                        enable_expert_tensor_parallelism=args.enable_expert_tensor_parallelism,
                    )
                    throwaway.set_deepspeed_parallelism()

                    ep_group = throwaway.deepspeed_moe.ep_group
                    if args.moe_token_dropping:
                        self.mlp = MbMoE(
                            neox_args, init_method, output_layer_init_method, ep_group
                        )
                    else:
                        self.mlp = dMoE(
                            neox_args, init_method, output_layer_init_method, ep_group
                        )

                    # Next, we trick DS into seeing these as its own MoE params.
                    for param in self.mlp.parameters():
                        if getattr(param, "expert_model_parallel", None) is not None:
                            # is_moe_param looks for this attr.
                            param.allreduce = False
                            param.group_name = throwaway.expert_group_name

                integrate_megablocks_with_ds_expert_parallelism()

            else:
                raise KeyError(neox_args.moe_type)

        self.layer_past = None  # used to cache k/v pairs in inference

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(self, x, attention_mask, layer_past=None, inter_position_ids=None, sep_atten_kernel_func=None):
        
        # ########################################################my SepLLM########################################################            
        if isinstance(attention_mask, (list, tuple)) or len(attention_mask.shape) > 4:
            attention_mask_layer = attention_mask[self.layer_number]
            if sep_atten_kernel_func is not None:
                sep_atten_kernel_func_layer = sep_atten_kernel_func[self.layer_number]
            else:
                sep_atten_kernel_func_layer = None            
        else:
            attention_mask_layer = attention_mask                    
            sep_atten_kernel_func_layer = sep_atten_kernel_func
            
        # ################################################my Debug###########################################
        # import os

        # torch.set_printoptions(profile="full")
        # print(f">>>>>>>>>>>>>>>>>>>attention_mask for layer {self.layer_number}:  RANK: {os.getenv('RANK')}:  {(~attention_mask_layer)[:,:,100,:]}.<<<<<<<<<<<<<<<<<<<")
        # torch.set_printoptions(profile="default")
        # ###################################################################################################
        # # #######################################################################################################################
        
        layer_past = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()
        moe_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # x: [b, s, h]
        if self.gpt_j_residual:  ##my:  True for pythia
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility

            residual = x
            # applies the correct normalization depending on if the norms are tied
            if self.gpt_j_tied: ##my:  False for pythia
                x = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            # attention operator
            attention_output, attention_bias = self.attention(
                x1, attention_mask_layer, layer_past=layer_past, inter_position_ids=inter_position_ids, sep_atten_kernel_func=sep_atten_kernel_func_layer
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output),
                    residual=None,
                    prob=self.hidden_dropout,
                )

            # mlp operator
            mlp_output, mlp_bias = self.mlp(x2)
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(mlp_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )

            # output = (x + attn(ln(x)) + mlp(ln(x))
            output = residual + self.reduce(output)
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            residual = x

            # x = x + attn(ln1(x))
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x), attention_mask_layer, layer_past=layer_past, inter_position_ids=inter_position_ids, sep_atten_kernel_func=sep_atten_kernel_func_layer
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents
            with torch.enable_grad():
                if attention_bias is not None:
                    # Use special bias_dropout_fn if we have a bias term from the above attention layer
                    attention_output = bias_dropout_fn(
                        attention_output,
                        bias=attention_bias.expand_as(residual),
                        residual=residual,
                        prob=self.hidden_dropout,
                    )
                else:
                    # Otherwise just apply dropout + residual
                    attention_output = (
                        torch.nn.functional.dropout(
                            attention_output,
                            p=self.hidden_dropout,
                            training=self.training,
                        )
                        + residual
                    )

            # output = x + mlp(ln2(x))
            layernorm_output = self.post_attention_layernorm(attention_output)
            mlp_bias = torch.tensor(
                0.0, device=layernorm_output.device, dtype=layernorm_output.dtype
            )

            if self.num_experts == 1:
                mlp_output, mlp_bias = self.mlp(layernorm_output)
            else:
                if self.moe_type == "deepspeed":
                    mlp_output, moe_loss, _ = self.mlp(layernorm_output)
                    mlp_bias = (
                        None  # deepspeed.moe.layer.MoE.forward ignores the bias term
                    )
                elif self.moe_type == "megablocks":
                    mlp_output, mlp_bias = self.mlp(layernorm_output)
                else:
                    raise KeyError(self.moe_type)

            with torch.enable_grad():
                if (
                    self.mlp_type == "llama"
                    or self.num_experts > 1
                    and self.moe_type == "deepspeed"
                ):
                    # No dropout either
                    assert mlp_bias is None
                    output = mlp_output + attention_output
                else:
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(attention_output),
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )

        return output, moe_loss


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):        

        inter_position_ids= None
        sep_atten_kernel_func = None   
        args = remove_none(args)     
        if self.USE_SEP_ATTN_KERNEL_ACCELERATOR:
            if self.USE_BiPE :       
                assert len(args) == 4, f"Incorrect number of arguments: ({len(args)}). ParallelTransformerLayerPipe expects 4 arguments If USE_SEP_ATTN_KERNEL_ACCELERATOR=True and USE_BiPE=True, i.e., (hidden_states, inter_position_ids, sep_atten_kernel_func, attention_mask)."
                hidden_states, inter_position_ids, sep_atten_kernel_func, attention_mask = args
                output, moe_loss = super().forward(hidden_states, attention_mask, inter_position_ids=inter_position_ids, sep_atten_kernel_func=sep_atten_kernel_func)
                self.last_moe_loss = moe_loss
                return output, inter_position_ids, sep_atten_kernel_func, attention_mask

            else:                                                
                assert len(args) == 3, f"Incorrect number of arguments: ({len(args)}). ParallelTransformerLayerPipe expects 3 arguments If USE_SEP_ATTN_KERNEL_ACCELERATOR=True and USE_BiPE=False, i.e.,  (hidden_states,  sep_atten_kernel_func,  attention_mask )."             
                hidden_states,  sep_atten_kernel_func,  attention_mask = args
                output, moe_loss = super().forward(hidden_states, attention_mask, inter_position_ids=inter_position_ids, sep_atten_kernel_func=sep_atten_kernel_func)
                self.last_moe_loss = moe_loss
                return output,  sep_atten_kernel_func, attention_mask

        else:
            if self.USE_BiPE :  
                assert len(args) == 3, f"Incorrect number of arguments: ({len(args)}). ParallelTransformerLayerPipe expects 3 arguments If USE_SEP_ATTN_KERNEL_ACCELERATOR=False and USE_BiPE=True, i.e., (hidden_states, inter_position_ids, attention_mask)."
                hidden_states, inter_position_ids, attention_mask = args
                output, moe_loss = super().forward(hidden_states, attention_mask, inter_position_ids=inter_position_ids, sep_atten_kernel_func=sep_atten_kernel_func)
                self.last_moe_loss = moe_loss
                return output, inter_position_ids, attention_mask

            else:        
                assert len(args) == 2, f"Incorrect number of arguments: ({len(args)}). ParallelTransformerLayerPipe expects 2 arguments If USE_SEP_ATTN_KERNEL_ACCELERATOR=False and USE_BiPE=False, i.e., (hidden_states, attention_mask)."                
                hidden_states, attention_mask = args                
                output, moe_loss = super().forward(hidden_states, attention_mask, inter_position_ids=inter_position_ids, sep_atten_kernel_func=sep_atten_kernel_func)
                self.last_moe_loss = moe_loss
                return output, attention_mask


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        logits, bias = super().forward(hidden_state)
        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.norm(args)


def parallel_lm_logits(
    input_,
    word_embeddings_weight,
    parallel_output,
    seq_parallel=False,
    seq_dim=1,
    bias=None,
):
    """LM logits using word embedding weights."""
    # Parallel logits.
    if seq_parallel:
        # if using Sequence Parallelism, our logits are sharded along the sequence dimension.
        # gather them here. (backward pass: reduce-scatter)
        input_parallel = mpu.gather_from_sequence_parallel_region(
            input_, seq_dim=seq_dim
        )
    else:
        # Set up backprop all-reduce.
        input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
