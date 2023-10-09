# FastCkpt: accelerate your LLM training in one line!

Fast gradient checkpoint is designed for accelerate the training with memory-efficient attention like FlashAttention and LightSeq. FastCkpt has monkey patch for both rematerialization-aware checkpointing and FlashAttention, so you can patch both in only one line!

Paper: https://arxiv.org/pdf/2310.03294.pdf

## News
- [2023/10] FastCkpt now supports LlamaModel in Huggingface!

## Install
```bash
pip install fastckpt
```

## Usage
FastCkpt now supports HF training pipeline. 

### Use FaskCkpt and FlashAttention
To use `fasckpt` with `flash_attn`, import and run `replace_hf_ckpt_with_fast_ckpt` *before* importing `transformers`
```python
# add monkey patch for fastckpt
from fastckpt.llama_flash_attn_ckpt_monkey_patch import replace_hf_ckpt_with_fast_ckpt
replace_hf_ckpt_with_fast_ckpt()

# import transformers and other packages
import transformers
...
```

### Use FlashAttention only
To only replace the `LlamaAttention` with `flash_attn` without chaning the checkpointing strategy, import and run `replace_llama_attn_with_flash_attn`

```python
# add monkey patch for fastckpt
from fastckpt.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

# import transformers and other packages
import transformers
...
```