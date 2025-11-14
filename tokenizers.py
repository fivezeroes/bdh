"""Tokenizer abstraction for BDH training.

This module provides a unified interface for different tokenization strategies,
allowing BDH to work with byte-level, TikToken, HuggingFace, and SentencePiece tokenizers.

Copyright Pathway Technology, Inc.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import warnings


class BaseTokenizer(ABC):
    """Base tokenizer interface."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        pass
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get padding token ID if available.
        
        Returns:
            Padding token ID or None
        """
        return None
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get end-of-sequence token ID if available.
        
        Returns:
            EOS token ID or None
        """
        return None
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Get beginning-of-sequence token ID if available.
        
        Returns:
            BOS token ID or None
        """
        return None


class ByteTokenizer(BaseTokenizer):
    """Byte-level tokenizer (current BDH implementation).
    
    Encodes text as UTF-8 bytes, treating each byte as a token.
    This is the default tokenizer for backward compatibility.
    """
    
    def encode(self, text: str) -> List[int]:
        """Encode text to byte-level token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of byte values (0-255)
        """
        return list(text.encode('utf-8', errors='ignore'))
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode byte-level token IDs to text.
        
        Args:
            token_ids: List of byte values (0-255)
            
        Returns:
            Decoded UTF-8 text
        """
        return bytes(token_ids).decode(errors='backslashreplace')
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (256 for all possible byte values).
        
        Returns:
            256
        """
        return 256
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID (null byte).
        
        Returns:
            0 (null byte)
        """
        return 0


class TikTokenTokenizer(BaseTokenizer):
    """TikToken tokenizer wrapper for GPT-style tokenization.
    
    Supports OpenAI's tokenizers like gpt2, cl100k_base (GPT-4), etc.
    Requires: pip install tiktoken
    """
    
    def __init__(self, encoding_name: str = "gpt2"):
        """Initialize TikToken tokenizer.
        
        Args:
            encoding_name: Name of the encoding (e.g., "gpt2", "cl100k_base")
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for TikToken tokenizer. "
                "Install with: pip install tiktoken"
            )
        
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        self._vocab_size = self.encoding.n_vocab
        
        # TikToken doesn't have explicit special tokens, but we can use EOT
        self._eos_token_id = self.encoding.eot_token
    
    def encode(self, text: str) -> List[int]:
        """Encode text using TikToken.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        return self.encoding.encode(text, allowed_special="all")
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs using TikToken.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text
        """
        return self.encoding.decode(token_ids)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size (e.g., 50257 for gpt2, 100277 for cl100k_base)
        """
        return self._vocab_size
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-text token ID.
        
        Returns:
            EOT token ID
        """
        return self._eos_token_id


class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace tokenizer wrapper.
    
    Supports any HuggingFace tokenizer from transformers library.
    Requires: pip install transformers
    """
    
    def __init__(self, model_name: str = "gpt2", **kwargs):
        """Initialize HuggingFace tokenizer.
        
        Args:
            model_name: HuggingFace model name or path to tokenizer
            **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained()
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace tokenizer. "
                "Install with: pip install transformers"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model_name = model_name
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(self, text: str) -> List[int]:
        """Encode text using HuggingFace tokenizer.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs using HuggingFace tokenizer.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size from tokenizer
        """
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get padding token ID.
        
        Returns:
            Padding token ID or None
        """
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get end-of-sequence token ID.
        
        Returns:
            EOS token ID or None
        """
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Get beginning-of-sequence token ID.
        
        Returns:
            BOS token ID or None
        """
        return self.tokenizer.bos_token_id


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece tokenizer wrapper.
    
    Supports custom SentencePiece models for multilingual tokenization.
    Requires: pip install sentencepiece
    """
    
    def __init__(self, model_path: str):
        """Initialize SentencePiece tokenizer.
        
        Args:
            model_path: Path to SentencePiece .model file
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece is required for SentencePiece tokenizer. "
                "Install with: pip install sentencepiece"
            )
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
    
    def encode(self, text: str) -> List[int]:
        """Encode text using SentencePiece.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        return self.sp.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs using SentencePiece.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text
        """
        return self.sp.decode(token_ids)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size from SentencePiece model
        """
        return self.sp.get_piece_size()
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID.
        
        Returns:
            Padding token ID (usually 0)
        """
        return self.sp.pad_id()
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID.
        
        Returns:
            EOS token ID
        """
        return self.sp.eos_id()
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID.
        
        Returns:
            BOS token ID
        """
        return self.sp.bos_id()


def create_tokenizer(tokenizer_type: str, tokenizer_name: Optional[str] = None, **kwargs) -> BaseTokenizer:
    """Factory function to create tokenizer instances.
    
    Args:
        tokenizer_type: Type of tokenizer ("byte", "tiktoken", "huggingface", "sentencepiece")
        tokenizer_name: Name or path for the tokenizer (required for non-byte tokenizers)
        **kwargs: Additional arguments passed to tokenizer constructor
        
    Returns:
        Initialized tokenizer instance
        
    Raises:
        ValueError: If tokenizer_type is unknown or tokenizer_name is missing when required
    """
    tokenizer_type = tokenizer_type.lower()
    
    if tokenizer_type == "byte":
        return ByteTokenizer()
    
    elif tokenizer_type == "tiktoken":
        if tokenizer_name is None:
            warnings.warn("No tokenizer_name provided for tiktoken, defaulting to 'gpt2'")
            tokenizer_name = "gpt2"
        return TikTokenTokenizer(encoding_name=tokenizer_name)
    
    elif tokenizer_type == "huggingface":
        if tokenizer_name is None:
            warnings.warn("No tokenizer_name provided for huggingface, defaulting to 'gpt2'")
            tokenizer_name = "gpt2"
        return HuggingFaceTokenizer(model_name=tokenizer_name, **kwargs)
    
    elif tokenizer_type == "sentencepiece":
        if tokenizer_name is None:
            raise ValueError("tokenizer_name (model path) is required for sentencepiece tokenizer")
        return SentencePieceTokenizer(model_path=tokenizer_name)
    
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Supported types: byte, tiktoken, huggingface, sentencepiece"
        )
