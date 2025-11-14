# Tokenization Guide for BDH

This guide explains how to use different tokenization strategies with the Baby Dragon Hatchling (BDH) training system.

## Overview

BDH now supports multiple tokenization options:

1. **Byte-level** (default) - Character-level encoding using UTF-8 bytes
2. **TikToken** - OpenAI's GPT-style tokenizers (requires `tiktoken`)
3. **HuggingFace** - Wide variety of pre-trained tokenizers (requires `transformers`)
4. **SentencePiece** - Custom trainable tokenizers (requires `sentencepiece`)

## Quick Start

### Using Byte-level Tokenization (Default)

No setup required! The default configuration uses byte-level tokenization:

```yaml
# config.yaml
tokenizer:
  type: "byte"
  name: null
```

**Pros:**
- No dependencies
- Language-agnostic
- Works with existing checkpoints
- Deterministic

**Cons:**
- Inefficient for long sequences (1 character ≈ 1 token)
- Doesn't capture linguistic structure
- Requires more training iterations

### Using TikToken (GPT-style)

1. Install the dependency:
```bash
pip install tiktoken
```

2. Update `config.yaml`:
```yaml
tokenizer:
  type: "tiktoken"
  name: "gpt2"  # or "cl100k_base" for GPT-4 tokenizer
```

**Available tokenizers:**
- `gpt2` - GPT-2 tokenizer (vocab_size: 50,257)
- `cl100k_base` - GPT-4 tokenizer (vocab_size: 100,277)

**Pros:**
- Fast Rust implementation
- 3-4x compression vs byte-level
- Good for English text
- Compatible with GPT models

**Cons:**
- Pre-trained vocabularies only
- Less flexible for custom domains

### Using HuggingFace Tokenizers

1. Install the dependency:
```bash
pip install transformers
```

2. Update `config.yaml`:
```yaml
tokenizer:
  type: "huggingface"
  name: "gpt2"  # or any HuggingFace model name
```

**Popular options:**
- `gpt2` - GPT-2 BPE tokenizer
- `bert-base-uncased` - BERT WordPiece tokenizer
- `facebook/opt-125m` - OPT tokenizer
- `meta-llama/Llama-2-7b-hf` - Llama tokenizer (requires access)

**Pros:**
- Extensive pre-trained options
- Supports many tokenizer types (BPE, WordPiece, etc.)
- Good ecosystem and documentation

**Cons:**
- Heavier dependency
- Can be slower than TikToken

### Using SentencePiece

1. Install the dependency:
```bash
pip install sentencepiece
```

2. Train your own tokenizer or use a pre-trained model file

3. Update `config.yaml`:
```yaml
tokenizer:
  type: "sentencepiece"
  name: "/path/to/your/model.model"  # Path to .model file
```

**Pros:**
- Language-agnostic
- Trainable on your data
- Good for multilingual models
- Supports subword regularization

**Cons:**
- Requires training or finding pre-trained models
- More complex setup

## Configuration Details

### Full Configuration Options

```yaml
tokenizer:
  type: "byte"  # "byte", "tiktoken", "huggingface", "sentencepiece"
  name: null  # Tokenizer name or path
  pad_token_id: null  # Auto-detected if null
  eos_token_id: null  # Auto-detected if null
  bos_token_id: null  # Auto-detected if null
```

### Vocabulary Size

The model's `vocab_size` is **automatically updated** from the tokenizer. You don't need to manually set it in the config.

For example, with `tiktoken` and `gpt2`:
```yaml
model:
  vocab_size: 256  # Will be auto-updated to 50257
```

## Training Considerations

### Block Size and Token Compression

Different tokenizers compress text differently:

- **Byte-level**: 512 characters ≈ 512 tokens
- **GPT-2 BPE**: 512 characters ≈ 128-256 tokens
- **GPT-4 tokenizer**: 512 characters ≈ 100-200 tokens

The `block_size` parameter refers to the number of **tokens**, not characters. With more efficient tokenizers, each batch covers more text.

### Memory Requirements

Larger vocabularies require more parameters:

| Tokenizer | Vocab Size | Embedding Params (n_embd=256) |
|-----------|------------|-------------------------------|
| Byte-level | 256 | 65,536 (~256 KB) |
| GPT-2 | 50,257 | 12,865,792 (~50 MB) |
| GPT-4 | 100,277 | 25,670,912 (~100 MB) |

### Checkpoint Compatibility

**Important:** Changing tokenizers requires retraining from scratch. Checkpoints saved with one tokenizer cannot be loaded with a different tokenizer type.

The tokenizer configuration is automatically saved in checkpoints and restored during inference.

## Inference with Tokenizers

### Using prompt.py

The inference script automatically loads the tokenizer from the checkpoint:

```bash
python prompt.py checkpoints/checkpoint_1000.pt --prompt "Hello world"
```

If the checkpoint has tokenizer info, it will be used automatically. Otherwise, it falls back to byte-level encoding.

### Interactive Mode

```bash
python prompt.py checkpoints/checkpoint_1000.pt
```

The interactive mode shows which tokenizer is active:

```
Interactive Prompting Mode
Tokenizer: TikTokenTokenizer
Settings: max_tokens=100, top_k=3, temperature=1.0
```

## Best Practices

### For English-only Models
Use **TikToken** with `gpt2` or `cl100k_base`:
```yaml
tokenizer:
  type: "tiktoken"
  name: "gpt2"
```

### For Multilingual Models
Use **SentencePiece** trained on your data:
```yaml
tokenizer:
  type: "sentencepiece"
  name: "/path/to/multilingual.model"
```

### For Maximum Compatibility
Stick with **byte-level** encoding:
```yaml
tokenizer:
  type: "byte"
```

### For Transfer Learning
Use **HuggingFace** tokenizer matching your base model:
```yaml
tokenizer:
  type: "huggingface"
  name: "gpt2"  # Match the architecture you're adapting
```

## Troubleshooting

### Import Errors

If you see errors like `Import "tiktoken" could not be resolved`, install the required dependency:

```bash
pip install tiktoken  # For TikToken
pip install transformers  # For HuggingFace
pip install sentencepiece  # For SentencePiece
```

### Vocabulary Size Mismatch

If loading a checkpoint fails with a vocabulary size error, ensure you're using the same tokenizer that was used during training.

### Slow Tokenization

If tokenization is slow:
- **TikToken** is fastest for GPT-style tokenization
- **HuggingFace** can be slower but has `use_fast=True` option
- **Byte-level** is simple and fast

### Memory Issues with Large Vocabularies

If you run out of memory with large vocabularies (50K+ tokens):
- Reduce `n_embd` in model config
- Use gradient checkpointing
- Enable 4-bit training via `low_precision.use_4bit: true`

## Migration Guide

### From Byte-level to BPE

1. **Create a new config** with the desired tokenizer
2. **Start training from scratch** (don't resume from byte-level checkpoints)
3. **Adjust hyperparameters** if needed (learning rate, block size, etc.)

Example:

```bash
# Old config (byte-level)
cp config.yaml config_byte.yaml

# Edit config.yaml for BPE
# Set tokenizer.type: "tiktoken"
# Set tokenizer.name: "gpt2"

# Start fresh training
python train.py --config config.yaml
```

### Testing Different Tokenizers

You can maintain multiple config files:

```bash
# Byte-level
python train.py --config config_byte.yaml

# GPT-2 tokenizer
python train.py --config config_gpt2.yaml

# Custom SentencePiece
python train.py --config config_sp.yaml
```

## Performance Comparison

Based on typical English text:

| Tokenizer | Compression | Speed | Memory | Best For |
|-----------|-------------|-------|--------|----------|
| Byte-level | 1.0x | Fast | Low | Small models, multilingual |
| GPT-2 BPE | 3-4x | Fastest | Medium | English, code |
| GPT-4 BPE | 4-5x | Fastest | High | English, efficient |
| HuggingFace | 3-4x | Medium | Medium | Flexibility, ecosystem |
| SentencePiece | 3-4x | Medium | Custom | Multilingual, custom domains |

## Advanced Usage

### Custom Special Tokens

You can override special token IDs in the config:

```yaml
tokenizer:
  type: "huggingface"
  name: "gpt2"
  pad_token_id: 50256  # Override default
  eos_token_id: 50256
  bos_token_id: null  # Don't use BOS token
```

### Tokenizer in Custom Scripts

You can use the tokenizer in your own scripts:

```python
from tokenizers import create_tokenizer

# Create tokenizer
tokenizer = create_tokenizer(
    tokenizer_type="tiktoken",
    tokenizer_name="gpt2"
)

# Encode text
tokens = tokenizer.encode("Hello, world!")
print(f"Tokens: {tokens}")

# Decode tokens
text = tokenizer.decode(tokens)
print(f"Text: {text}")

# Get vocab size
print(f"Vocab size: {tokenizer.vocab_size}")
```

## Support

For issues or questions about tokenization:
1. Check this documentation
2. Review the error messages (they're designed to be helpful)
3. Verify your tokenizer dependencies are installed
4. Ensure your config matches your checkpoint's tokenizer

## References

- TikToken: https://github.com/openai/tiktoken
- HuggingFace Tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer
- SentencePiece: https://github.com/google/sentencepiece
- BPE Paper: https://arxiv.org/abs/1508.07909
