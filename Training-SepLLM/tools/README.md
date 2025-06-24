# (SepLLM) GPT-NeoX Auxiliary Tools

This directory contains a number of auxiliary tools that are useful for working with (SepLLM) GPT-NeoX but not part of the main training code.

## Bash

This directory `general_bash` contains some simple, frequently used bash commands to make working on multiple machines easier.

## Checkpoints

This directory `ckpts` contains tools for manipulating and converting checkpoints including changing the parallelism settings of a pretrained model, converting between (SepLLM) GPT-NeoX and the transformers library, and updating checkpoints trained with Version 1.x of this library to be compatible with Version 2.x.

## Datasets

This directory `datasets` contains tools for downloading and preprocessing datasets to the format expected by the (SepLLM) GPT-NeoX library.

## SepLLM

This directory `sepllm` contains tools for obtaining the token ids for the separator tokens.

# Note
Most tools are inherited directly from the GPT-NeoX codebase.
