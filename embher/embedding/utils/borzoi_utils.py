"""
Borzoi model utilities.

- Model initialization (AnnotatedBorzoi, or Flashzoi for faster inference)
- Sequence processing (numba-accelerated one-hot encoding)
- Track index search

Flashzoi is a FlashAttention-2 optimized version of Borzoi (~1.6x faster inference,
~18% less memory). Use `use_flashzoi=True` in `init_borzoi_model()`.
"""

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any

from .gpu import select_device

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    from borzoi_pytorch import Borzoi, AnnotatedBorzoi
except ImportError:
    Borzoi = None
    AnnotatedBorzoi = None


# =============================================================================
# Fast One-Hot Encoding (Numba-accelerated)
# Credit: tangermeme library (https://github.com/jmschrei/tangermeme)
# =============================================================================

if HAS_NUMBA:
    @numba.njit("void(int8[:, :], int8[:], int8[:])")
    def _fast_one_hot_encode(X_ohe, seq, mapping):
        """JIT-compiled one-hot encoding inner loop."""
        for i in range(len(seq)):
            idx = mapping[seq[i]]
            if idx == -1:  # ignored character (N)
                continue
            if idx == -2:
                raise ValueError("Encountered unknown character")
            X_ohe[i, idx] = 1


def _onehot_encoding_numba(sequence: str, alphabet: str = "ACGT", 
                           dtype: torch.dtype = torch.float32, 
                           ignore: str = "N",
                           neutral_value: float = 0.0) -> torch.Tensor:
    """
    Fast one-hot encoding using numba JIT compilation.
    
    Args:
        sequence: DNA sequence string (will be uppercased)
        alphabet: Characters to encode (default: "ACGT")
        dtype: Output tensor dtype (default: torch.float32)
        ignore: Characters to ignore (default: "N")
        neutral_value: Value to fill for ignored characters (default: 0.0).
                       Use 0.25 for uniform distribution across bases.
        
    Returns:
        Tensor of shape (len(alphabet), len(sequence))
    """
    if not HAS_NUMBA:
        raise ImportError("numba is required for fast one-hot encoding. "
                         "Install with: pip install numba")
    
    seq_idxs = np.frombuffer(bytearray(sequence.upper(), "utf8"), dtype=np.int8)
    alpha_idxs = np.frombuffer(bytearray(alphabet, "utf8"), dtype=np.int8)
    ignore_idxs = np.frombuffer(bytearray(ignore, "utf8"), dtype=np.int8)

    one_hot_mapping = np.zeros(256, dtype=np.int8) - 2
    for i, idx in enumerate(alpha_idxs):
        one_hot_mapping[idx] = i
    for idx in ignore_idxs:
        one_hot_mapping[idx] = -1

    one_hot = np.zeros((len(sequence), len(alphabet)), dtype=np.int8)
    _fast_one_hot_encode(one_hot, seq_idxs, one_hot_mapping)
    
    # Convert to float and apply neutral_value for ignored characters
    one_hot_float = one_hot.astype(np.float32)
    if neutral_value != 0.0:
        # Find positions where all values are zero (ignored characters)
        ignored_mask = one_hot_float.sum(axis=1) == 0
        one_hot_float[ignored_mask, :] = neutral_value
    
    return torch.from_numpy(one_hot_float).type(dtype).T


# =============================================================================
# Borzoi Model Constants
# =============================================================================

BORZOI_SEQ_LEN = 524288       # Input sequence length (524kb)
BORZOI_BIN_SIZE = 32          # Output bin size in bp
BORZOI_NUM_BINS = 6144        # Number of output bins
BORZOI_NUM_TRACKS = 7611      # Total number of output tracks
BORZOI_PRED_LENGTH = 196608   # Prediction coverage: 6144 * 32 bp (~196kb)

# Supported context window sizes for Borzoi inference
# Smaller windows = faster VCF parsing, but sequences are center-padded with 'N'
# Note: The model always receives 524KB input; smaller context windows are padded
BORZOI_SUPPORTED_CONTEXT_WINDOWS = [
    4096,    # 4KB
    8192,    # 8KB
    16384,   # 16KB
    65536,   # 64KB
    102400,  # 100KB
    131072,  # 128KB
    262144,  # 256KB
    524288,  # 524KB (full, default)
]

# Model name mappings for different replicates
BORZOI_MODEL_NAMES = {
    0: 'johahi/borzoi-replicate-0',
    1: 'johahi/borzoi-replicate-1',
    2: 'johahi/borzoi-replicate-2',
    3: 'johahi/borzoi-replicate-3',
}

FLASHZOI_MODEL_NAMES = {
    0: 'johahi/flashzoi-replicate-0',
    1: 'johahi/flashzoi-replicate-1',
    2: 'johahi/flashzoi-replicate-2',
    3: 'johahi/flashzoi-replicate-3',
}

# Default model names
DEFAULT_BORZOI_MODEL = 'johahi/borzoi-replicate-0'
DEFAULT_FLASHZOI_MODEL = 'johahi/flashzoi-replicate-0'


def get_model_name(
    replicate: int = 0,
    use_flashzoi: bool = False
) -> str:
    """
    Get the HuggingFace model name for a Borzoi/Flashzoi replicate.
    
    Args:
        replicate: Model replicate number (0-3)
        use_flashzoi: If True, return Flashzoi model name (FlashAttention-2 optimized)
        
    Returns:
        HuggingFace model name string
        
    Example:
        >>> get_model_name(0, use_flashzoi=False)
        'johahi/borzoi-replicate-0'
        >>> get_model_name(0, use_flashzoi=True)
        'johahi/flashzoi-replicate-0'
    """
    if replicate not in range(4):
        raise ValueError(f"replicate must be 0-3, got {replicate}")
    
    if use_flashzoi:
        return FLASHZOI_MODEL_NAMES[replicate]
    return BORZOI_MODEL_NAMES[replicate]


def _materialize_annotation_buffers(model: Any) -> None:
    """
    Rebuild AnnotatedBorzoi's annotation buffers if they were left on `meta`.

    AnnotatedBorzoi.__init__ registers scale_values/clip_values/track_transform
    with `.to(self.conv_dna.conv_layer.weight.device)`. Under transformers >=4.5x
    from_pretrained builds the module on the `meta` device first, so that device
    is `meta`; because the buffers are persistent=False they are not in the
    checkpoint and are never materialized, and the later .to(device) fails with
    "Cannot copy out of meta tensor". Re-running the builder once the real
    weights are in place recreates them on the right device.
    """
    meta = [n for n, b in model.named_buffers() if b.is_meta]
    if not meta:
        return
    model._build_annotation_df(model.tracks_df)
    still_meta = [n for n, b in model.named_buffers() if b.is_meta]
    if still_meta:
        raise RuntimeError(f"Borzoi buffers still on meta device: {still_meta}")


def init_borzoi_model(
    model_name: Optional[str] = None,
    device: str = 'auto',
    use_flashzoi: bool = False,
    replicate: int = 0,
    use_random_init: bool = False
) -> Tuple[Any, Optional[pd.DataFrame]]:
    """
    Initialize Borzoi or Flashzoi model and load track metadata.

    Args:
        model_name: HuggingFace model name (if provided, overrides use_flashzoi and replicate)
        device: Device to load model on. Options:
                - 'auto': Automatically select best GPU (most free memory) or CPU
                - 'cuda': Use default CUDA device
                - 'cuda:N': Use specific GPU N
                - 'cpu': Use CPU
        use_flashzoi: If True, use Flashzoi (FlashAttention-2 optimized) for faster inference.
                      Provides ~1.6x speedup and ~18% memory reduction.
        replicate: Model replicate number (0-3), used when model_name is None
        use_random_init: If True, initialize model with random weights instead of loading
                         pretrained weights. Useful for baseline comparisons.

    Returns:
        Tuple of (model, tracks_df)
        - model: Loaded Borzoi/Flashzoi model (eval mode)
        - tracks_df: DataFrame with track metadata (if available)
        
    Example:
        >>> # Load standard Borzoi (auto-selects best GPU)
        >>> model, tracks_df = init_borzoi_model()
        
        >>> # Load Flashzoi for faster inference
        >>> model, tracks_df = init_borzoi_model(use_flashzoi=True)
        
        >>> # Load specific replicate
        >>> model, tracks_df = init_borzoi_model(use_flashzoi=True, replicate=1)
        
        >>> # Load specific model by name
        >>> model, tracks_df = init_borzoi_model(model_name='johahi/flashzoi-replicate-0')
        
        >>> # Force CPU
        >>> model, tracks_df = init_borzoi_model(device='cpu')

        >>> # Random initialization (no pretrained weights)
        >>> model, tracks_df = init_borzoi_model(use_flashzoi=True, use_random_init=True)

    Note:
        Flashzoi uses FlashAttention-2 which requires:
        - CUDA GPU with compute capability >= 8.0 (Ampere or newer)
        - flash-attn package installed
        
        Performance comparison (on NVIDIA RTX A4000):
        - Standard Borzoi: ~0.24s per inference, ~3.5 GB memory
        - Flashzoi: ~0.15s per inference, ~2.9 GB memory
    """
    if AnnotatedBorzoi is None:
        raise ImportError("borzoi_pytorch not installed. Please install it to use Borzoi models.")

    # Auto-select device if requested
    if device == 'auto':
        device = select_device(prefer_cuda=True)
        print(f"Auto-selected device: {device}")

    # Handle random initialization (no pretrained weights)
    if use_random_init:
        return _init_borzoi_random(device, use_flashzoi, replicate)

    # Determine model name
    if model_name is None:
        model_name = get_model_name(replicate=replicate, use_flashzoi=use_flashzoi)
    
    # Determine model variant for logging
    is_flashzoi = 'flashzoi' in model_name.lower()
    variant_name = "Flashzoi" if is_flashzoi else "Borzoi"
    
    try:
        # Try loading AnnotatedBorzoi first to get metadata
        model = AnnotatedBorzoi.from_pretrained(model_name)
        _materialize_annotation_buffers(model)
        tracks_df = model.tracks_df.copy()
        print(f"Loaded {variant_name} model (AnnotatedBorzoi) with {len(tracks_df)} tracks")
        if is_flashzoi:
            print("  → Using FlashAttention-2 for faster inference (~1.6x speedup)")
    except Exception as e:
        print(f"Could not load AnnotatedBorzoi ({e}), falling back to Borzoi class")
        model = Borzoi.from_pretrained(model_name)
        tracks_df = None
        print(f"Loaded {variant_name} model (Borzoi) without track metadata")
        if is_flashzoi:
            print("  → Using FlashAttention-2 for faster inference (~1.6x speedup)")

    model = model.to(device)
    model.eval()

    return model, tracks_df


def _init_borzoi_random(
    device: str,
    use_flashzoi: bool,
    replicate: int
) -> Tuple[Any, Optional[pd.DataFrame]]:
    """
    Initialize Borzoi/Flashzoi with random weights (no pretrained loading).

    This creates the model architecture with random PyTorch initialization,
    without downloading any pretrained weights from HuggingFace.

    Args:
        device: Device to load model on
        use_flashzoi: If True, use FlashAttention-2 architecture
        replicate: Model replicate number (for track metadata URL)

    Returns:
        Tuple of (model, tracks_df)
    """
    from borzoi_pytorch.config_borzoi import BorzoiConfig

    # Create config - set flashed=True for FlashAttention-2
    config = BorzoiConfig(
        flashed=use_flashzoi,  # True = FlashAttention, False = regular attention
        enable_mouse_head=True,  # Match pretrained config
    )

    # Direct instantiation - NO weights loaded!
    model = Borzoi(config)
    model = model.to(device)
    model.eval()

    variant = "Flashzoi" if use_flashzoi else "Borzoi"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Initialized RANDOM {variant} (no pretrained weights)")
    print(f"  flashed={use_flashzoi}, params={n_params:.1f}M")

    # Load tracks_df from borzoi_pytorch package (no network required)
    tracks_df = None
    try:
        from borzoi_pytorch.pytorch_borzoi_model import TRACKS_DF
        tracks_df = TRACKS_DF.copy()
        print(f"Loaded track metadata ({len(tracks_df)} tracks)")
    except Exception as e:
        print(f"Could not load track metadata: {e}")

    return model, tracks_df


def dna_to_onehot(
    sequence: str,
    seq_len: int = 524288,
    neutral_value: float = 0.0
) -> torch.Tensor:
    """
    Convert DNA sequence string to one-hot encoded tensor for Borzoi.
    
    Uses numba-accelerated encoding for fast performance.
    
    Args:
        sequence: DNA sequence string
        seq_len: Target sequence length (default 524288 for Borzoi)
        neutral_value: Value to fill for N bases (default: 0.0).
                       Use 0.25 for uniform distribution across bases.
        
    Returns:
        Tensor of shape (1, 4, seq_len)
        
    Raises:
        ImportError: If numba is not installed
    """
    if not HAS_NUMBA:
        raise ImportError("numba is required for one-hot encoding. "
                         "Install with: pip install numba")
    
    # Pad or truncate to target length
    current_len = len(sequence)
    if current_len < seq_len:
        # Center pad with N
        pad_left = (seq_len - current_len) // 2
        pad_right = seq_len - current_len - pad_left
        sequence = 'N' * pad_left + sequence + 'N' * pad_right
    elif current_len > seq_len:
        # Center crop
        start = (current_len - seq_len) // 2
        sequence = sequence[start : start + seq_len]
    
    one_hot = _onehot_encoding_numba(sequence, dtype=torch.float32, neutral_value=neutral_value)
    return one_hot.unsqueeze(0)  # (1, 4, L)


def get_borzoi_track_indices(
    tracks_df: pd.DataFrame,
    description_query: str,
    assay_type: Optional[str] = None
) -> List[int]:
    """
    Find track indices matching a description query.

    Args:
        tracks_df: DataFrame with track metadata
        description_query: String to search for in description (case-insensitive)
        assay_type: Optional filter for assay type (e.g., 'DNASE', 'RNA', 'CAGE')

    Returns:
        List of track indices
    """
    if tracks_df is None:
        return []
        
    mask = tracks_df['description'].str.contains(description_query, case=False, na=False)
    
    if assay_type:
        # Assay type is typically at the beginning of description (e.g., "RNA:", "CAGE:", "DNASE:")
        if 'assay' in tracks_df.columns:
            mask &= (tracks_df['assay'] == assay_type)
        else:
            # Check if description starts with assay type prefix
            assay_prefix = f"{assay_type}:"
            mask &= tracks_df['description'].str.startswith(assay_prefix, na=False)
            
    return tracks_df[mask].index.tolist()
