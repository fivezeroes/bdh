"""View tokenizer vocabulary.

This script displays the vocabulary/dictionary used by different tokenizers.
"""

from tokenizers import create_tokenizer, HuggingFaceTokenizer, SentencePieceTokenizer
from config import Config
import argparse


def view_byte_tokenizer_vocab():
    """View ByteTokenizer vocabulary (all 256 bytes)."""
    print("ByteTokenizer Vocabulary (256 bytes):")
    print("=" * 60)
    for i in range(256):
        try:
            char = bytes([i]).decode('utf-8')
            if char.isprintable():
                print(f"Token {i:3d}: '{char}' (0x{i:02x})")
            else:
                print(f"Token {i:3d}: <non-printable> (0x{i:02x})")
        except:
            print(f"Token {i:3d}: <invalid> (0x{i:02x})")


def view_tiktoken_vocab(encoding_name="gpt2", sample_size=100):
    """View TikToken vocabulary (sample)."""
    tokenizer = create_tokenizer("tiktoken", encoding_name)
    print(f"TikToken Vocabulary ({encoding_name}):")
    print(f"Total vocab size: {tokenizer.vocab_size}")
    print(f"Showing first {sample_size} tokens:")
    print("=" * 60)
    
    for i in range(min(sample_size, tokenizer.vocab_size)):
        try:
            decoded = tokenizer.decode([i])
            print(f"Token {i:5d}: {repr(decoded)}")
        except:
            print(f"Token {i:5d}: <error decoding>")


def view_huggingface_vocab(model_name="gpt2", sample_size=100):
    """View HuggingFace tokenizer vocabulary."""
    tokenizer = create_tokenizer("huggingface", model_name)
    print(f"HuggingFace Tokenizer Vocabulary ({model_name}):")
    print(f"Total vocab size: {tokenizer.vocab_size}")
    print("=" * 60)
    
    # Access the underlying tokenizer's vocabulary
    if isinstance(tokenizer, HuggingFaceTokenizer):
        vocab = tokenizer.tokenizer.get_vocab()
        print(f"\nShowing first {sample_size} tokens (sorted by ID):")
        
        # Sort by token ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab[:sample_size]:
            print(f"Token {token_id:5d}: {repr(token)}")
        
        # Show special tokens
        print("\n" + "=" * 60)
        print("Special Tokens:")
        print(f"  PAD token: {tokenizer.tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  BOS token: {tokenizer.tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"  EOS token: {tokenizer.tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")


def view_sentencepiece_vocab(model_path, sample_size=100):
    """View SentencePiece tokenizer vocabulary."""
    tokenizer = create_tokenizer("sentencepiece", model_path)
    print(f"SentencePiece Tokenizer Vocabulary ({model_path}):")
    print(f"Total vocab size: {tokenizer.vocab_size}")
    print(f"Showing first {sample_size} tokens:")
    print("=" * 60)
    
    if isinstance(tokenizer, SentencePieceTokenizer):
        for i in range(min(sample_size, tokenizer.vocab_size)):
            piece = tokenizer.sp.id_to_piece(i)
            print(f"Token {i:5d}: {repr(piece)}")


def view_current_tokenizer_vocab(config_path="config.yaml", sample_size=100):
    """View vocabulary of the tokenizer configured in config.yaml."""
    config = Config.from_yaml(config_path)
    
    tokenizer_type = config.tokenizer.type
    tokenizer_name = config.tokenizer.name
    
    print(f"Current Configuration:")
    print(f"  Tokenizer Type: {tokenizer_type}")
    print(f"  Tokenizer Name: {tokenizer_name}")
    print("\n")
    
    if tokenizer_type == "byte":
        view_byte_tokenizer_vocab()
    elif tokenizer_type == "tiktoken":
        view_tiktoken_vocab(tokenizer_name or "gpt2", sample_size)
    elif tokenizer_type == "huggingface":
        view_huggingface_vocab(tokenizer_name or "gpt2", sample_size)
    elif tokenizer_type == "sentencepiece":
        if tokenizer_name:
            view_sentencepiece_vocab(tokenizer_name, sample_size)
        else:
            print("Error: tokenizer_name (model path) required for SentencePiece")
    else:
        print(f"Unknown tokenizer type: {tokenizer_type}")


def main():
    parser = argparse.ArgumentParser(description="View tokenizer vocabulary")
    parser.add_argument(
        '--type',
        choices=['byte', 'tiktoken', 'huggingface', 'sentencepiece', 'config'],
        default='config',
        help='Tokenizer type to view (default: use config.yaml)'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Tokenizer name/model (e.g., "gpt2", "cl100k_base")'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of tokens to display (default: 100, use -1 for all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    if args.type == 'config':
        view_current_tokenizer_vocab(args.config, args.sample_size)
    elif args.type == 'byte':
        view_byte_tokenizer_vocab()
    elif args.type == 'tiktoken':
        view_tiktoken_vocab(args.name or "gpt2", args.sample_size)
    elif args.type == 'huggingface':
        view_huggingface_vocab(args.name or "gpt2", args.sample_size)
    elif args.type == 'sentencepiece':
        if not args.name:
            print("Error: --name (model path) required for SentencePiece")
            return
        view_sentencepiece_vocab(args.name, args.sample_size)


if __name__ == "__main__":
    main()
