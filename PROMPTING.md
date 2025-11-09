# Prompting Guide for BDH

This guide shows you how to use the `prompt.py` script to load trained BDH checkpoints and generate text.

## Basic Usage

### Interactive Mode (Default)

The easiest way to experiment with your trained model:

```bash
python prompt.py checkpoints/checkpoint_1000.pt
```

This launches an interactive session where you can:
- Type any prompt and press Enter to generate text
- Type `settings` to adjust generation parameters
- Type `quit` or `exit` to stop

### Single Prompt Mode

Generate text from a single prompt and exit:

```bash
python prompt.py checkpoints/checkpoint_1000.pt --prompt "The Capital of France is"
```

## Generation Parameters

You can customize how the model generates text:

```bash
python prompt.py checkpoints/checkpoint_1000.pt \
    --prompt "Once upon a time" \
    --max-tokens 200 \
    --top-k 5 \
    --temperature 0.8
```

### Parameter Guide

- **`--max-tokens`**: Maximum number of new tokens to generate (default: 100)
  - Higher values = longer completions
  - Example: `--max-tokens 200`

- **`--top-k`**: Limits sampling to the top K most probable tokens (default: 3)
  - Lower values (1-3) = more focused/deterministic output
  - Higher values (10+) = more diverse output
  - Example: `--top-k 10`

- **`--temperature`**: Controls randomness in sampling (default: 1.0)
  - Lower values (0.1-0.7) = more conservative/predictable
  - 1.0 = standard sampling
  - Higher values (1.2+) = more creative/random
  - Example: `--temperature 0.7`

## Interactive Mode Commands

When running in interactive mode:

1. **Enter a prompt**: Just type your text and press Enter
   ```
   Prompt: The meaning of life is
   ```

2. **Change settings**: Type `settings` and follow the prompts
   ```
   Prompt: settings
   Max new tokens (current: 100): 200
   Top-k (current: 3): 5
   Temperature (current: 1.0): 0.8
   Updated settings: max_tokens=200, top_k=5, temperature=0.8
   ```

3. **Exit**: Type `quit`, `exit`, or press Ctrl+C
   ```
   Prompt: quit
   Goodbye!
   ```

## Examples

### Example 1: Creative Writing

```bash
python prompt.py checkpoints/checkpoint_5000.pt \
    --prompt "In a world where dragons" \
    --max-tokens 300 \
    --top-k 10 \
    --temperature 1.2
```

### Example 2: Factual Completion

```bash
python prompt.py checkpoints/checkpoint_5000.pt \
    --prompt "The three branches of government are" \
    --max-tokens 100 \
    --top-k 3 \
    --temperature 0.5
```

### Example 3: Code Generation

```bash
python prompt.py checkpoints/checkpoint_10000.pt \
    --prompt "def fibonacci(n):" \
    --max-tokens 150 \
    --top-k 5 \
    --temperature 0.7
```

## Tips

1. **Start with default parameters** and adjust based on output quality
2. **Lower temperature** for more predictable/factual responses
3. **Higher temperature** for creative or diverse outputs
4. **Smaller top-k** for focused, coherent text
5. **Larger top-k** for more varied vocabulary
6. **Use interactive mode** to quickly experiment with different prompts
7. **Model quality depends on training** - later checkpoints are typically better

## Troubleshooting

**Q: Model generates gibberish**
- Try a checkpoint from later in training
- Lower the temperature (try 0.7)
- Reduce top-k (try 3)

**Q: Model is too repetitive**
- Increase temperature (try 1.2)
- Increase top-k (try 10+)

**Q: Checkpoint file not found**
- Check that the path is correct
- Make sure you've trained the model first with `train.py`
- List available checkpoints: `ls checkpoints/`

**Q: Out of memory**
- Reduce max-tokens
- The model should work on CPU/MPS if CUDA isn't available
