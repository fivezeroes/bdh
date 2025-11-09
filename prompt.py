# Copyright Pathway Technology, Inc.

import os
import sys
import argparse
from contextlib import nullcontext

import bdh
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

dtype = (
    "bfloat16"
    if (device.type == "cuda" and torch.cuda.is_bf16_supported()) or device.type == "mps"
    else "float16"
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)

print(f"Using device: {device} with dtype {dtype}")


def load_checkpoint_for_inference(checkpoint_path):
    """Load model checkpoint for inference."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint if available, otherwise use default
    config = checkpoint.get('config', bdh.BDHConfig())
    
    # Create model with the config
    model = bdh.BDH(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    step = checkpoint.get('step', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    print(f"Loaded checkpoint from step {step} (loss: {loss})")
    
    return model


def generate_text(model, prompt_text, max_new_tokens=100, top_k=3, temperature=1.0):
    """Generate text from a prompt."""
    # Convert prompt to tensor
    prompt_bytes = prompt_text.encode('utf-8', errors='ignore')
    prompt_tensor = torch.tensor(
        bytearray(prompt_bytes), dtype=torch.long, device=device
    ).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            prompt_tensor, 
            max_new_tokens=max_new_tokens, 
            top_k=top_k,
            temperature=temperature
        )
    
    # Decode output
    output_bytes = bytes(output.to(torch.uint8).to("cpu").squeeze(0))
    output_text = output_bytes.decode(errors='backslashreplace')
    
    return output_text


def interactive_mode(model, max_new_tokens=100, top_k=3, temperature=1.0):
    """Run interactive prompting mode."""
    print("\n" + "="*60)
    print("Interactive Prompting Mode")
    print("="*60)
    print(f"Settings: max_tokens={max_new_tokens}, top_k={top_k}, temperature={temperature}")
    print("Type your prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'settings' to change generation parameters.")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("\nPrompt: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if prompt.lower() == 'settings':
                try:
                    new_max_tokens = input(f"Max new tokens (current: {max_new_tokens}): ").strip()
                    if new_max_tokens:
                        max_new_tokens = int(new_max_tokens)
                    
                    new_top_k = input(f"Top-k (current: {top_k}): ").strip()
                    if new_top_k:
                        top_k = int(new_top_k)
                    
                    new_temp = input(f"Temperature (current: {temperature}): ").strip()
                    if new_temp:
                        temperature = float(new_temp)
                    
                    print(f"Updated settings: max_tokens={max_new_tokens}, top_k={top_k}, temperature={temperature}")
                except ValueError as e:
                    print(f"Invalid input: {e}")
                continue
            
            if not prompt:
                print("Empty prompt. Please enter some text.")
                continue
            
            print("\nGenerating...")
            output = generate_text(model, prompt, max_new_tokens, top_k, temperature)
            print("\n" + "-"*60)
            print(output)
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Load a checkpoint and prompt the model')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt to generate from (non-interactive)')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum new tokens to generate (default: 100)')
    parser.add_argument('--top-k', type=int, default=3, help='Top-k sampling parameter (default: 3)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode (default if no --prompt)')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_checkpoint_for_inference(args.checkpoint)
    
    # Decide mode
    if args.prompt is not None:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...")
        output = generate_text(model, args.prompt, args.max_tokens, args.top_k, args.temperature)
        print("\n" + "="*60)
        print(output)
        print("="*60)
    else:
        # Interactive mode (default)
        interactive_mode(model, args.max_tokens, args.top_k, args.temperature)


if __name__ == "__main__":
    main()
