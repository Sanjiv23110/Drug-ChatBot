"""
Azure OpenAI embedder with proper TPM throttling.

Prevents rate limit errors and ensures stable ingestion.
"""
import time
import numpy as np
from typing import List
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class AzureEmbedder:
    """
    Azure OpenAI embedder with token-per-minute (TPM) throttling.
    
    Tracks token usage in sliding 60-second window and sleeps
    if approaching limit.
    """
    
    def __init__(
        self,
        tpm_limit: int = 400_000,
        model: str = "text-embedding-3-small",
        dimension: int = 1536
    ):
        """
        Initialize embedder with rate limiting.
        
        Args:
            tpm_limit: Tokens per minute limit (default 150k)
            model: Embedding model name
            dimension: Embedding dimension
        """
        self.tpm_limit = tpm_limit
        self.model = model
        self.dimension = dimension
        
        # Track token usage: [(timestamp, token_count), ...]
        self.tokens_used_window = []
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count from text.
        
        Rough estimate: 1 token ≈ 4 characters
        This is conservative (actual is ~3.5 for English).
        """
        return max(1, len(text) // 4)
    
    def _get_current_tpm(self) -> int:
        """Get tokens used in last 60 seconds."""
        now = time.time()
        
        # Remove entries older than 60 seconds
        self.tokens_used_window = [
            (ts, tokens) for ts, tokens in self.tokens_used_window
            if now - ts < 60
        ]
        
        # Sum remaining tokens
        return sum(tokens for _, tokens in self.tokens_used_window)
    
    def _wait_if_needed(self, token_count: int):
        """
        Sleep if adding tokens would exceed TPM limit.
        
        Args:
            token_count: Tokens about to be used
        """
        current_tpm = self._get_current_tpm()
        
        if current_tpm + token_count > self.tpm_limit:
            # Calculate sleep time
            oldest_ts = self.tokens_used_window[0][0]
            sleep_time = 60 - (time.time() - oldest_ts)
            
            if sleep_time > 0:
                print(f"TPM limit approaching ({current_tpm}/{self.tpm_limit}), sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                
                # Clear window after sleep
                self.tokens_used_window = []
    
    def embed_batch(
        self,
        texts: List[str],
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Embed a batch of texts with retry logic.
        
        Args:
            texts: List of texts to embed (max 50 recommended)
            max_retries: Number of retry attempts on failure
            
        Returns:
            (N, dimension) numpy array of embeddings
            
        Raises:
            Exception: If all retries fail
        """
        if not texts:
            return np.array([])
        
        # Estimate total tokens
        total_tokens = sum(self.estimate_tokens(t) for t in texts)
        
        # Wait if needed
        self._wait_if_needed(total_tokens)
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                # Call Azure OpenAI
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.deployment_name
                )
                
                # Extract embeddings
                embeddings = np.array([e.embedding for e in response.data])
                
                # Validate dimension
                if embeddings.shape[1] != self.dimension:
                    raise ValueError(
                        f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}"
                    )
                
                # Record token usage
                self.tokens_used_window.append((time.time(), total_tokens))
                
                return embeddings
            
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    raise Exception(f"Embedding failed after {max_retries} attempts: {e}")
                
                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"Embedding attempt {attempt + 1} failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        # Should never reach here
        raise Exception("Embedding failed unexpectedly")
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            (dimension,) numpy array
        """
        embeddings = self.embed_batch([text])
        return embeddings[0]
