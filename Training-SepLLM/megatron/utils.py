# Copyright (c) 2024, EleutherAI
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

"""General utilities."""
import os
import sys
import re
import time
import socket
from typing import Dict, List

import requests

try:
    import wandb
except ModuleNotFoundError:
    pass

import torch

from deepspeed.launcher.runner import fetch_hostfile, parse_inclusion_exclusion
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

from megatron import print_rank_0
from megatron import mpu

from collections import deque
from packaging.version import Version


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()
    return reduced_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(
        torch.cuda.max_memory_allocated() / mega_bytes
    )
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(
        torch.cuda.max_memory_reserved() / mega_bytes
    )
    print_rank_0(string)


def get_attn_mask(seq_length, device, sliding_window_width):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )
    # get rid of lower diagonals than the sliding window width, if a value was provided
    if sliding_window_width is not None:
        mask = torch.triu(mask, diagonal=-sliding_window_width)

    # convert to binary
    return mask < 0.5


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    eod_mask_loss=False,
    sliding_window_width=None,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    attention_mask = get_attn_mask(
        seq_length=seq_length,
        device=data.device,
        sliding_window_width=sliding_window_width,
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids


def local_rank():
    """Local rank of process"""
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")

    if local_rank is None:
        print(
            "utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0",
            flush=True,
        )
        local_rank = 0
    return int(local_rank)


def is_bnb_available():
    """True if bitsandbytes optimizers are available"""
    return importlib.util.find_spec("bitsandbytes") is not None


def is_local_main():
    """True if is the local main process"""
    return local_rank() == 0


def is_mp_rank_0():
    """True if mp rank == 0"""
    return mpu.get_model_parallel_rank() == 0


def get_wandb_api_key(neox_args):
    """Get Weights and Biases API key from ENV or .netrc file. Otherwise return None"""
    if "WANDB_LOCAL" in os.environ:
        return "LOCAL"
    if "WANDB_API_KEY" in os.environ:
        return os.environ["WANDB_API_KEY"]

    wandb_token = requests.utils.get_netrc_auth(neox_args.wandb_host)

    if wandb_token is not None:
        return wandb_token[1]


def init_wandb(neox_args):
    # Wandb. (one worker per machine)
    if neox_args.use_wandb == False:
        return

    if not neox_args.wandb_init_all_ranks:
        use_wandb = is_local_main() and (
            get_wandb_api_key(neox_args=neox_args) is not None
        )
        neox_args.update_value("use_wandb", use_wandb)
    if neox_args.use_wandb:
        group_name = neox_args.wandb_group
        name = f"{socket.gethostname()}-{local_rank()}" if group_name else None
        try:
            wandb.init(
                project=neox_args.wandb_project,
                group=group_name,
                name=name,
                save_code=False,
                force=False,
                entity=neox_args.wandb_team,
            )
        except wandb.UsageError as e:
            neox_args.update_value("use_wandb", False)
            print(e)
            print(
                "Skipping wandb. Execute `wandb login` on local or main node machine to enable.",
                flush=True,
            )
        wandb.config.update(neox_args.all_config)


def obtain_resource_pool(
    hostfile_path, include_arg, exclude_arg
) -> Dict[str, List[int]]:
    """
    Get dict of `resource_pool[hostname] = [list of GPU ranks]` using hostfile, include and exclude args.
    Modified from: `deepspeed.launcher.runner.main`
    """
    resource_pool = fetch_hostfile(hostfile_path)
    if not resource_pool:
        resource_pool = {}
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool["localhost"] = device_count

    active_resources = parse_inclusion_exclusion(
        resource_pool, include_arg, exclude_arg
    )
    return active_resources


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def ddb(rank=0):
    """
    Distributed Debugger that will insert a py debugger on rank `rank` and
    pause all other distributed processes until debugging is complete.
    :param rank:
    """
    if torch.distributed.get_rank() == rank:
        from pdb import Pdb

        pdb = Pdb(skip=["torch.distributed.*"])
        pdb.set_trace(sys._getframe().f_back)
    torch.distributed.barrier()


class Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # pollutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


def expand_attention_types(attention_config, num_layers):
    """
    Expands an `attention_config` list in the following format:

        [
        [['attention_type_1', ..., `attention_type_n`], 12]
        ]

    to a flattened list of length `num_layers`.

    :param params_list:
    :return:
    """
    # if only strings are found in the config, we assume it's already expanded
    if all([isinstance(i, str) for i in attention_config]):
        return attention_config
    newlist = []
    for item in attention_config:
        # instead of specifying a number - we can specify 'all' to extend this pattern across all layers
        if item[1] == "all":
            assert num_layers % len(item[0]) == 0, (
                f"Number of layers ({num_layers}) is not divisible by the length "
                f"of pattern: {item[0]}"
            )
            return item[0] * (num_layers // len(item[0]))
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist


class OverflowMonitor:

    """
    Checks if the past n iterations have been skipped due to overflow, and exits
    training if that happens.
    """

    def __init__(self, optimizer, n=50):
        self.optimizer = optimizer
        self.n = n
        self.history = deque(maxlen=n)
        self.bf16 = isinstance(optimizer, BF16_Optimizer)

    def check(self, skipped):
        if self.bf16:
            return
        self.history.append(skipped)
        if (
            self.optimizer.overflow
            and len(self.history) == self.n
            and all(self.history)
        ):
            raise Exception(
                f"Skipped {self.n} iterations in a row due to Overflow - Exiting training."
            )


def get_noise_scale_logger(neox_args):
    if neox_args.log_gradient_noise_scale:
        if neox_args.zero_stage >= 1:
            raise NotImplementedError(
                "Gradient Noise Scale logging does not work with zero stage 2+, as the "
                "gradients are distributed across ranks."
            )
        noise_scale_logger = GradientNoiseScale(
            model=model,
            batch_size_small=neox_args.train_batch_size,
            n_batches=neox_args.gradient_noise_scale_n_batches,
            cpu_offload=neox_args.gradient_noise_scale_cpu_offload,
            neox_args=neox_args,
            mpu=mpu,
        )
    else:
        noise_scale_logger = None
    return noise_scale_logger


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        print(
            " > number of parameters on model parallel rank {}: {}".format(
                mpu.get_model_parallel_rank(), params
            ),
            flush=True,
        )
    else:
        params = 0

    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())
    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def setup_for_inference_or_eval(use_cache=True, overwrite_values=None, input_args=None):
    """
    Initializes the model for evaluation or inference (doesn't load optimizer states, etc.) from command line args.

    use_cache: bool
        Whether to use key value caching in inference.
    overwrite_values: dict
        Optional Values to overwrite in the model config.
    """

    from megatron.neox_arguments import NeoXArgs
    from megatron.initialize import initialize_megatron
    from megatron.training import setup_model_and_optimizer

    _overwrite_values = {
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        "optimizer": None,  # prevent loading optimizer (no_load_optim alone won't work)
        "zero_optimization": None,  # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
    }
    if overwrite_values:
        _overwrite_values.update(overwrite_values)
    neox_args = NeoXArgs.consume_neox_args(
        overwrite_values=_overwrite_values, input_args=input_args
    )
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()

    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize wandb
    init_wandb(neox_args=neox_args)

    # initialize megatron
    initialize_megatron(neox_args)

    # set up model and load checkpoint.
    model, _, _ = setup_model_and_optimizer(
        neox_args=neox_args,
        use_cache=use_cache,
        iteration=neox_args.iteration,
    )  # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed
    print_rank_0("Finished loading model")

    model.module.inference_mode(use_cache=use_cache)
    return model, neox_args


class CharCounter:
    """
    Wraps the data_iterator to count the number of characters in a batch
    """

    def __init__(self, data_iterator, tokenizer):
        self.tokenizer = tokenizer
        self.data_iterator = data_iterator
        self.char_count = 0
        self.batch_count = 0
        self.token_count = 0
        self.total_time = 0

    def tokens_per_char(self):
        return self.token_count / self.char_count

    def __iter__(self):
        return self

    def __next__(self):
        start = time.time()
        batch = self.data_iterator.__next__()
        for b in batch["text"]:
            self.token_count += len(b)
            self.char_count += len(self.tokenizer.detokenize(b.tolist()))
        self.batch_count += 1
        end = time.time()
        self.total_time += end - start
        return batch


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


class SepLLMArgumentsChecker:
    
    def __init__(self, neox_args=None):
        self.neox_args = neox_args


    def check_list_items_equal(self, a_list, target=None):
        assert isinstance(a_list, list), f"`a_list` must be a list"
        for el in a_list:
            if el != target:
                return False
        return True

    def check_list_items_general(self, a_list, criteria_func):
        assert isinstance(a_list, list), f"`a_list` must be a list"
        for el in a_list:
            if not criteria_func(el):
                return False
        return True        


    def set_list_items(self, src_list, target=None):
        assert isinstance(src_list, list), f"`src_list` must be a list"
        return [ target for i in range(len(src_list))]


    def __call__(self, neox_args, STRICT_CHECK=True):        

        print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        print(">>>>>>>>---------               Start Checking Arguments                   -----------<<<<<<<<")                
        print(">>>>>>>>---------##########################################################-----------<<<<<<<<\n\n")        

        EXPERIMENT_NUM = int(neox_args.USE_ORIGINAL_FULL_ATTEN) + int(neox_args.streamingLLM) + int(neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR) + int(neox_args.USE_SA_SOFTMAX) + int(neox_args.USE_SA_SOFTMAX_NO_DENO)
        UNIQUE_EXP_FLAG = EXPERIMENT_NUM <= 1        
        assert UNIQUE_EXP_FLAG, f"You can only set at most one True among ('USE_ORIGINAL_FULL_ATTEN'={neox_args.USE_ORIGINAL_FULL_ATTEN}, 'streamingLLM'={neox_args.streamingLLM}, 'USE_SEP_ATTN_KERNEL_ACCELERATOR'={neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR}, 'USE_SA_SOFTMAX'={neox_args.USE_SA_SOFTMAX}, and 'USE_SA_SOFTMAX_NO_DENO'={neox_args.USE_SA_SOFTMAX_NO_DENO}) at one time"        

        changed_param_num = 0
        if neox_args.USE_ORIGINAL_FULL_ATTEN:
            print(f"'USE_ORIGINAL_FULL_ATTEN' is the signal flag with the HIGHEST PRIORITY, i.e., setting it to True will modify certain related hyperparameters to run the standard full-attention version of the model. Details are as follows:")
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------               USE_ORIGINAL_FULL_ATTEN                    -----------<<<<<<<<")
            print(">>>>>>>>--------- Running the original full attention LLM (no changing).--------------<<<<<<<<")
            print(">>>>>>>>--------- All SepLLM-related settings will NOT take any effect     -----------<<<<<<<<")            
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<\n\n")


            print("***********************************************************************************************")
            print("*****************************************Warnings**********************************************")
            print(f">>'USE_ORIGINAL_FULL_ATTEN' is the signal flag with the HIGHEST PRIORITY, i.e., setting it<<**")
            print(f">> to True will modify certain related hyperparameters to run the standard full-attention <<**")
            print(f">> Details are as follows:                                                                <<**")            
                                    
            if (neox_args.separator_token_ids is not None) and (neox_args.separator_token_ids != [-1]):                
                print(f">> separator_token_ids:{neox_args.separator_token_ids}----changed to---->   {[-1]}   ---disabled <<**")
                neox_args.separator_token_ids = [-1]       
                changed_param_num += 1     
            if neox_args.PADDING_ID != 0 :
                print(f">> PADDING_ID:{neox_args.PADDING_ID}----changed to---->   {0} <<**")
                neox_args.PADDING_ID = 0       
                changed_param_num += 1      


            if neox_args.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS:
                print(f">> USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS:{neox_args.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS}----changed to---->   {False}   ---disabled <<**")
                neox_args.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS = False
                changed_param_num += 1   
            if neox_args.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS:
                print(f">> USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS:{neox_args.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS}----changed to---->   {False}   ---disabled <<**")
                neox_args.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS = False
                changed_param_num += 1   
            if neox_args.prefill_local_window_size != neox_args.seq_length:
                print(f">> prefill_local_window_size:{neox_args.prefill_local_window_size}----changed to---->   seq_length=={neox_args.seq_length} <<**")
                neox_args.prefill_local_window_size = neox_args.seq_length
                changed_param_num += 1   
            if neox_args.generate_local_window_size != neox_args.seq_length:
                print(f">> generate_local_window_size:{neox_args.generate_local_window_size}----changed to---->   seq_length=={neox_args.seq_length} <<**")
                neox_args.generate_local_window_size = neox_args.seq_length
                changed_param_num += 1   

            if neox_args.init_tok_max_idx != -1:
                print(f">> init_tok_max_idx:{neox_args.init_tok_max_idx}----changed to---->   {-1}   ---disabled <<**")
                neox_args.init_tok_max_idx = -1
                changed_param_num += 1   

            if neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR:
                print(f">> USE_SEP_ATTN_KERNEL_ACCELERATOR:{neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR}----changed to---->   {False}   ---disabled <<**")
                neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR = False
                changed_param_num += 1 
   
            if neox_args.RECOMPILE_SEP_ATTN_KERNEL:
                print(f">> RECOMPILE_SEP_ATTN_KERNEL:{neox_args.RECOMPILE_SEP_ATTN_KERNEL}----changed to---->   {False}   ---disabled <<**")
                neox_args.RECOMPILE_SEP_ATTN_KERNEL = False
                changed_param_num += 1    

            if neox_args.USE_BiPE:
                print(f">> USE_BiPE:{neox_args.USE_BiPE}----changed to---->   {False}   ---disabled <<**")
                neox_args.USE_BiPE = False
                changed_param_num += 1 
            if neox_args.BiPE_seps:
                print(f">> BiPE_seps:{neox_args.BiPE_seps}----changed to---->   {None}   ---disabled <<**")
                neox_args.BiPE_seps = None
                changed_param_num += 1 

            if neox_args.BATCH_ADAPTIVE_INIT_POS:
                print(f">> BATCH_ADAPTIVE_INIT_POS:{neox_args.BATCH_ADAPTIVE_INIT_POS}----changed to---->   {False}   ---disabled <<**")
                neox_args.BATCH_ADAPTIVE_INIT_POS = False
                changed_param_num += 1    

            if neox_args.PRINT_KV_RATIO:
                print(f">> PRINT_KV_RATIO:{neox_args.PRINT_KV_RATIO}----changed to---->   {False}   ---disabled;trivial <<**")
                neox_args.PRINT_KV_RATIO = False
                changed_param_num += 1    

            if STRICT_CHECK:
                if not self.check_list_items_equal(neox_args.attention_config, 'global'):
                    print(f">> attention_config:{neox_args.attention_config}----changed to---->   {['global'] * len(neox_args.attention_config) }   <<**")
                    neox_args.attention_config = self.set_list_items(neox_args.attention_config, 'global')
                    changed_param_num += 1   

            # print(f"attention-config:  {neox_args.attention_config}, type: {type(neox_args.attention_config)}")

            if changed_param_num > 0:
                print(f"*****************************Total {changed_param_num} hyperparameters are reset*****************************")
            else:
                print(f"***************************No hyperparameters need to be reset********************************")

        elif neox_args.streamingLLM:
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                          -----------<<<<<<<<")
            print(">>>>>>>>------------------------ Running streamingLLM --------------------------------<<<<<<<<")
            print(">>>>>>>>---------Only settings about local and initial sizes take effect.  -----------<<<<<<<<") 
            print(">>>>>>>>---------##########################################################-----------<<<<<<<<\n\n")


            print("***********************************************************************************************")
            print("*****************************************Warnings**********************************************")
            print(f">> To run 'streamingLLM', we need to disable 'separator_token_ids'.Details are as follows: <<**")               

            if neox_args.separator_token_ids != [-1] :                
                print(f">> separator_token_ids:{neox_args.separator_token_ids}----changed to---->   {[-1]}   ---disabled <<**")
                neox_args.separator_token_ids = [-1]
                changed_param_num += 1   
            else:
                print(f">> separator_token_ids:{neox_args.separator_token_ids} has already been disabled <<**")
            
            assert not neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR, f"To run streamingLLM, must set USE_SEP_ATTN_KERNEL_ACCELERATOR to False"
            assert not neox_args.USE_BiPE, f"To run streamingLLM, must set USE_BiPE to False"

            assert neox_args.init_tok_max_idx >= 0, f"To run streamingLLM, must set init_tok_max_idx to some value that is greater than (or equal to) 0"

            assert not neox_args.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS, f"To run streamingLLM, must set USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS to False"
            assert not neox_args.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS, f"To run streamingLLM, must set USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS to False"

            assert self.check_list_items_equal(neox_args.attention_config, 'global'), f'streamingLLM relies on attention-config=[[["global"], <num-layers>]] '

        elif neox_args.USE_SA_SOFTMAX or neox_args.USE_SA_SOFTMAX_NO_DENO:
            assert self.check_list_items_equal(neox_args.attention_config, 'global'), f'USE_SA_SOFTMAX=True or USE_SA_SOFTMAX_NO_DENO=True rely on attention-config=[[["global"], <num-layers>]] '

        else:
            if neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR:
                assert Version(torch.__version__) >= Version('2.5.0'), f"If you want to use the Sep_Attention kernel module (USE_SEP_ATTN_KERNEL_ACCELERATOR=True) to accelerate the training process of SepLLM, the torch version must be at least 2.5.0 or newer, but got {torch.__version__}. Please set USE_SEP_ATTN_KERNEL_ACCELERATOR=False to run SepLLM, which is logically the same but slower (the speed is similar to full attention)"

            # assert self.check_list_items_equal(neox_args.attention_config, 'global'), f'SepLLM relies on attention-config=[[["global"], <num-layers>]] '
            
            assert isinstance(neox_args.separator_token_ids, list) and (neox_args.separator_token_ids != [-1]) and len(neox_args.separator_token_ids)>=1, f"To run SepLLM, separator_token_ids should not be None or [-1], and must be well set."

        assert neox_args.PADDING_ID == 0, f"PADDING_ID must be 0 for Pythia or GPTNeox" 
        assert neox_args.generate_local_window_size >= int(STRICT_CHECK), f"generate_local_window_size should be greater than (or equal to) {int(STRICT_CHECK)}." 
        assert neox_args.prefill_local_window_size >= int(STRICT_CHECK), f"prefill_local_window_size should be greater than (or equal to) {int(STRICT_CHECK)}." 

        if neox_args.RECOMPILE_SEP_ATTN_KERNEL:
            assert neox_args.USE_SEP_ATTN_KERNEL_ACCELERATOR, f"RECOMPILE_SEP_ATTN_KERNEL=True only takes effect when USE_SEP_ATTN_KERNEL_ACCELERATOR=True and may require additional GPU memory while offering a certain degree of acceleration to the training process."

        if neox_args.generate_local_window_size != neox_args.prefill_local_window_size:
            print(f"Warnings: It is recommended to set the value of generate_local_window_size={neox_args.generate_local_window_size} to be the same as prefill_local_window_size={neox_args.prefill_local_window_size}, even though generate_local_window_size does not have any effect during the pretraining/prefilling phase.")
        if neox_args.USE_PREFILL_LOCAL_WIN_SIZES_wrt_LAYERS:
            assert self.check_list_items_general(neox_args.prefill_loc_win_size_list, lambda x: x>=int(STRICT_CHECK))
        if neox_args.USE_GENERATE_LOCAL_WIN_SIZES_wrt_LAYERS:
            assert self.check_list_items_general(neox_args.generate_win_loc_size_list, lambda x: x>=int(STRICT_CHECK))
            print(f"Warnings: It is recommended to set the value of generate_win_loc_size_list={neox_args.generate_win_loc_size_list} to be the same as prefill_loc_win_size_list={neox_args.prefill_loc_win_size_list}, even though generate_win_loc_size_list does not have any effect during the pretraining/prefilling phase.")

        assert neox_args.init_tok_max_idx >= -1, f"init_tok_max_idx={neox_args.init_tok_max_idx} should be greater than (or equal to) -1. init_tok_max_idx=-1 means 'disabled' "

        if neox_args.USE_BiPE:
            assert isinstance(neox_args.BiPE_seps, list) and neox_args.BiPE_seps != [-1] and len(neox_args.BiPE_seps)>=1, f"If USE_BiPE=True, you should well set BiPE_seps (which cannot be None or [-1]). For exmaple, BiPE_seps=separator_token_ids={neox_args.separator_token_ids}"
            assert not neox_args.rope_fusion, "If you want to use BiPE, you cannot use rope_fusion simultaneously"

        if neox_args.BATCH_ADAPTIVE_INIT_POS:
            print(f"Warnings: During pretraining, it is recommended to set the value of BATCH_ADAPTIVE_INIT_POS to False since pretraining uses right padding.")


        print(">>>>>>>>---------##########################################################-----------<<<<<<<<")
        print(">>>>>>>>---------                 Initial Check Finished                   -----------<<<<<<<<")                
        print(">>>>>>>>---------##########################################################-----------<<<<<<<<\n\n")      

        return neox_args


