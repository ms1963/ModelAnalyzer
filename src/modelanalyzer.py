#!/usr/bin/env python3
"""
ModelAnalyzer v2.0.2 - COMPLETE FIXED VERSION
==============================================
Complete production-ready model analysis tool with all fixes applied.

Author:  Michael Stal, 2026
License: MIT
"""

import argparse
import json
import sys
import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

# Check required dependencies
try:
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import HfApi, model_info
except ImportError:
    print("❌ Missing required dependencies!")
    print("   Install: pip install transformers huggingface-hub")
    sys.exit(1)

# Optional dependencies
MATPLOTLIB_AVAILABLE = False
TQDM_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Version &amp; Constants
# ============================================================================

__version__ = "2.0.2"

# Network settings
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2
RATE_LIMIT_DELAY = 0.5

# Model limits
MAX_REASONABLE_CONTEXT = 1_000_000
MAX_REASONABLE_VOCAB = 1_000_000
MAX_REASONABLE_PARAMS = 1_000_000_000_000
MAX_REASONABLE_LAYERS = 1000
MAX_REASONABLE_HIDDEN = 100_000
MIN_REASONABLE_HIDDEN = 1

# Memory constants
DEFAULT_SEQ_LENGTH = 2048
DEFAULT_BATCH_SIZE = 1
MB_TO_GB = 1024

BYTES_PER_PARAM = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'int8': 1,
    'int4': 0.5,
    'int2': 0.25
}

OPTIMIZER_MEMORY_MULTIPLIER = {
    'adam': 2.0,
    'sgd': 1.0,
    'adamw': 2.0,
}

# File size limits
MAX_JSON_SIZE_MB = 100
MAX_MARKDOWN_SIZE_MB = 50
MAX_CONFIG_SIZE_ITEMS = 1000
MAX_JSON_RECURSION_DEPTH = 100
LARGE_MODEL_WARNING_MB = 1000

# Error messages
ERROR_MESSAGES = {
    'invalid_model_id': "Invalid model ID format",
    'config_load_failed': "Failed to load model configuration",
    'tokenizer_load_failed': "Failed to load tokenizer",
    'network_timeout': "Network request timed out",
    'auth_failed': "Authentication failed",
    'path_invalid': "Invalid output path",
    'disk_full': "Insufficient disk space",
    'rate_limited': "Rate limited by API",
}


# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Setup logging configuration."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if verbose:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter('%(message)s')
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


# ============================================================================
# Enums
# ============================================================================

class QuantizationType(Enum):
    """Types of quantization."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    GGUF_Q4_K_M = "gguf-q4-k-m"
    GGUF_Q5_K_S = "gguf-q5-k-s"
    GGUF_Q6_K = "gguf-q6-k"
    GGUF_Q8_0 = "gguf-q8-0"
    BNB_4BIT = "bnb-4bit"
    BNB_8BIT = "bnb-8bit"
    GGML = "ggml"
    EXLLAMA = "exllama"
    EXL2 = "exl2"
    SQUEEZELLM = "squeezellm"
    MIXED = "mixed"


class ModelFamily(Enum):
    """Known model families - ORDER MATTERS (most specific first)."""
    CODELLAMA = "codellama"
    LLAMA3 = "llama3"
    LLAMA2 = "llama2"
    LLAMA = "llama"
    MIXTRAL = "mixtral"
    MISTRAL = "mistral"
    QWEN2 = "qwen2"
    QWEN = "qwen"
    BLOOMZ = "bloomz"
    BLOOM = "bloom"
    GPT_NEOX = "gpt-neox"
    GPT_NEO = "gpt-neo"
    GPT_J = "gpt-j"
    GPT2 = "gpt2"
    STARCODER = "starcoder"
    SANTACODER = "santacoder"
    CODEGEN = "codegen"
    OPT = "opt"
    PYTHIA = "pythia"
    FALCON = "falcon"
    MPT = "mpt"
    STABLELM_EPOCH = "stablelm-epoch"
    STABLELM = "stablelm"
    REDPAJAMA = "redpajama"
    BAICHUAN = "baichuan"
    INTERNLM = "internlm"
    CHATGLM = "chatglm"
    PHI3 = "phi-3"
    PHI2 = "phi-2"
    PHI = "phi"
    GEMMA = "gemma"
    YI = "yi"
    DEEPSEEK = "deepseek"
    SOLAR = "solar"
    BERT = "bert"
    ROBERTA = "roberta"
    ALBERT = "albert"
    ELECTRA = "electra"
    T5 = "t5"
    BART = "bart"
    PEGASUS = "pegasus"
    CLIP = "clip"
    VIT = "vit"
    DEIT = "deit"
    SWIN = "swin"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Model type classification."""
    CAUSAL_LM = "causal-lm"
    MASKED_LM = "masked-lm"
    SEQ2SEQ = "seq2seq"
    ENCODER_ONLY = "encoder-only"
    DECODER_ONLY = "decoder-only"
    ENCODER_DECODER = "encoder-decoder"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


class TrainingObjective(Enum):
    """Training objectives."""
    BASE = "base"
    INSTRUCT = "instruct"
    CHAT = "chat"
    CODE = "code"
    MATH = "math"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QA = "question-answering"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    REWARD = "reward"
    UNKNOWN = "unknown"


class AttentionType(Enum):
    """Attention mechanism types."""
    STANDARD = "standard"
    GROUPED_QUERY = "grouped-query"
    MULTI_QUERY = "multi-query"
    SLIDING_WINDOW = "sliding-window"
    FLASH = "flash-attention"
    SPARSE = "sparse"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_token: Optional[int] = None
    expert_capacity: Optional[int] = None
    router_type: Optional[str] = None
    expert_hidden_size: Optional[int] = None
    shared_expert: bool = False
    num_shared_experts: Optional[int] = None


@dataclass
class AttentionConfig:
    """Attention mechanism configuration."""
    attention_type: AttentionType = AttentionType.STANDARD
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    sliding_window_size: Optional[int] = None
    use_flash_attention: bool = False
    use_alibi: bool = False
    use_rope: bool = False
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None


@dataclass
class ArchitectureDetails:
    """Detailed architecture information."""
    architecture_type: str = "unknown"
    num_layers: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    vocab_size: Optional[int] = None
    
    # Encoder-decoder specific
    encoder_layers: Optional[int] = None
    decoder_layers: Optional[int] = None
    encoder_hidden_size: Optional[int] = None
    decoder_hidden_size: Optional[int] = None
    
    # Vision specific
    image_size: Optional[int] = None
    patch_size: Optional[int] = None
    num_channels: Optional[int] = None
    
    # Advanced features
    use_cache: bool = True
    tie_word_embeddings: bool = False
    is_encoder_decoder: bool = False
    
    # Normalization
    layer_norm_type: Optional[str] = None
    layer_norm_eps: Optional[float] = None
    rms_norm_eps: Optional[float] = None
    
    # Activation
    activation_function: Optional[str] = None
    hidden_act: Optional[str] = None
    
    # Dropout
    attention_dropout: Optional[float] = None
    hidden_dropout: Optional[float] = None
    residual_dropout: Optional[float] = None
    
    # Bias
    use_bias: bool = True
    attention_bias: bool = True
    
    # Additional config
    additional_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizerInfo:
    """Tokenizer information."""
    tokenizer_type: Optional[str] = None
    vocab_size: Optional[int] = None
    model_max_length: Optional[int] = None
    
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    unk_token: Optional[str] = None
    sep_token: Optional[str] = None
    pad_token: Optional[str] = None
    cls_token: Optional[str] = None
    mask_token: Optional[str] = None
    
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    additional_special_tokens: List[str] = field(default_factory=list)
    
    do_lower_case: bool = False
    add_prefix_space: bool = False
    trim_offsets: bool = True
    
    chat_template: Optional[str] = None
    has_chat_template: bool = False


@dataclass
class QuantizationInfo:
    """Quantization information."""
    quantization_type: QuantizationType = QuantizationType.NONE
    bits_per_param: float = 32.0
    is_quantized: bool = False
    
    quantization_config: Optional[Dict[str, Any]] = None
    weight_dtype: Optional[str] = None
    compute_dtype: Optional[str] = None
    
    gptq_bits: Optional[int] = None
    gptq_group_size: Optional[int] = None
    gptq_desc_act: bool = False
    
    awq_bits: Optional[int] = None
    awq_group_size: Optional[int] = None
    awq_version: Optional[str] = None
    
    bnb_4bit_compute_dtype: Optional[str] = None
    bnb_4bit_quant_type: Optional[str] = None
    bnb_4bit_use_double_quant: bool = False


@dataclass
class MemoryEstimate:
    """Memory usage estimates."""
    params_fp32_mb: float = 0.0
    params_fp16_mb: float = 0.0
    params_int8_mb: float = 0.0
    params_int4_mb: float = 0.0
    
    activation_memory_mb: float = 0.0
    
    kv_cache_per_token_mb: float = 0.0
    kv_cache_full_context_mb: float = 0.0
    
    total_inference_fp16_mb: float = 0.0
    total_inference_int8_mb: float = 0.0
    total_training_fp32_mb: float = 0.0
    total_training_fp32_adam_mb: float = 0.0


@dataclass
class ModelCapabilities:
    """Model capabilities."""
    can_generate: bool = False
    supports_beam_search: bool = False
    supports_sampling: bool = False
    
    max_context_length: Optional[int] = None
    supports_long_context: bool = False
    
    supports_streaming: bool = False
    supports_batching: bool = True
    supports_gradient_checkpointing: bool = False
    
    is_trainable: bool = True
    supports_lora: bool = False
    supports_qlora: bool = False
    
    supports_flash_attention: bool = False
    supports_torch_compile: bool = False
    supports_bettertransformer: bool = False


@dataclass
class ModelMetadata:
    """Model metadata."""
    model_id: str
    author: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    tags: List[str] = field(default_factory=list)
    library_name: Optional[str] = None
    pipeline_tag: Optional[str] = None
    license: Optional[str] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    
    model_size_mb: Optional[float] = None
    safetensors_size_mb: Optional[float] = None
    
    is_gated: bool = False
    gated_reason: Optional[str] = None


@dataclass
class ComprehensiveModelInfo:
    """Complete model information."""
    model_id: str
    model_family: ModelFamily
    model_type: ModelType
    training_objective: TrainingObjective
    
    architecture: ArchitectureDetails
    attention: AttentionConfig
    moe: MoEConfig
    
    tokenizer: TokenizerInfo
    quantization: QuantizationInfo
    
    num_parameters: int
    num_parameters_human: str
    trainable_parameters: int
    
    memory: MemoryEstimate
    capabilities: ModelCapabilities
    metadata: ModelMetadata
    
    raw_config: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Helper Functions
# ============================================================================

def validate_model_id(model_id: str) -> Tuple[bool, Optional[str]]:
    """Validate model ID format."""
    if not model_id or not isinstance(model_id, str):
        return False, "Model ID must be a non-empty string"
    
    model_id = model_id.strip()
    if not model_id:
        return False, "Model ID cannot be empty"
    
    if '..' in model_id or model_id.startswith('/'):
        return False, "Model ID contains invalid path characters"
    
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
    for char in invalid_chars:
        if char in model_id:
            return False, f"Model ID contains invalid character: {char}"
    
    if len(model_id) > 500:
        return False, "Model ID too long (max 500 characters)"
    
    return True, None


def validate_output_path(path: str) -> Tuple[bool, Optional[str]]:
    """Validate output file path with symlink protection."""
    try:
        path_obj = Path(path).resolve()
        
        # Check for symlink attacks
        if path_obj.is_symlink():
            real_path = path_obj.resolve()
            if not str(real_path).startswith(str(Path.cwd())):
                return False, "Symlink points outside working directory"
        
        try:
            path_obj.relative_to(Path.cwd())
        except ValueError:
            if not path_obj.is_absolute():
                return False, "Relative path outside current directory"
        
        parent = path_obj.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create directory: {e}"
        
        if path_obj.exists() and not os.access(path, os.W_OK):
            return False, "No write permission"
        
        if not path_obj.exists() and not os.access(parent, os.W_OK):
            return False, "No write permission for directory"
        
        try:
            stat = os.statvfs(parent)
            free_space_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            if free_space_mb < 10:
                return False, "Insufficient disk space (< 10MB)"
        except:
            pass
        
        return True, None
    except Exception as e:
        return False, f"Invalid path: {e}"


def safe_getattr(obj: Any, *attrs: str, default: Any = None) -> Any:
    """Safely get nested attributes."""
    if obj is None:
        return default
    
    for attr in attrs:
        value = getattr(obj, attr, None)
        if value is not None:
            return value
    return default


def safe_config_to_dict(config: Any, max_items: int = MAX_CONFIG_SIZE_ITEMS) -> Dict[str, Any]:
    """Safely convert config to dictionary with size limit."""
    if config is None:
        return {}
    
    try:
        if hasattr(config, 'to_dict'):
            result = config.to_dict()
        elif hasattr(config, '__dict__'):
            result = dict(config.__dict__)
        else:
            return {}
        
        # Limit size
        if len(result) > max_items:
            logger.debug(f"Config too large ({len(result)} items), truncating")
            result = dict(list(result.items())[:max_items])
            result['_truncated'] = True
        
        return result
    except Exception as e:
        logger.debug(f"Could not convert config to dict: {e}")
        return {}


def clamp(value: Union[int, float, None], min_val: Union[int, float], 
          max_val: Union[int, float]) -> Union[int, float]:
    """Clamp value between min and max."""
    if value is None:
        return min_val
    return max(min_val, min(value, max_val))


def validate_positive_int(value: Optional[int], name: str = "value",
                         min_val: int = 1, max_val: Optional[int] = None,
                         allow_zero: bool = False) -> Optional[int]:
    """Validate positive integer with bounds."""
    if value is None:
        return None
    
    try:
        value = int(value)
        
        if allow_zero and value == 0:
            return 0
        
        if value < min_val:
            logger.debug(f"{name} below minimum ({min_val}): {value}")
            return None
        if max_val is not None and value > max_val:
            logger.debug(f"{name} exceeds maximum ({max_val}): {value}")
            return None
        return value
    except (ValueError, TypeError):
        logger.debug(f"{name} not a valid integer: {value}")
        return None


def validate_float_range(value: Optional[float], name: str = "value",
                        min_val: float = 0.0, max_val: float = 1.0) -> Optional[float]:
    """Validate float within range."""
    if value is None:
        return None
    
    try:
        value = float(value)
        if value < min_val or value > max_val:
            logger.debug(f"{name} outside range [{min_val}, {max_val}]: {value}")
            return None
        return value
    except (ValueError, TypeError):
        logger.debug(f"{name} not a valid float: {value}")
        return None


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, delay: float = RATE_LIMIT_DELAY):
        self.delay = delay
        self.last_call = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()


_rate_limiter = RateLimiter()


def retry_with_backoff(func: Callable, max_retries: int = MAX_RETRIES, 
                       delay: float = RETRY_DELAY, timeout: int = DEFAULT_TIMEOUT,
                       *args, **kwargs) -> Any:
    """Retry function with exponential backoff and timeout."""
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func)}")
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            _rate_limiter.wait()
            
            if 'timeout' in func.__code__.co_varnames:
                kwargs['timeout'] = timeout
            
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            if 'rate limit' in error_str or '429' in error_str:
                logger.warning(ERROR_MESSAGES['rate_limited'])
                wait_time = delay * (2 ** (attempt + 2))
            else:
                wait_time = delay * (2 ** attempt)
            
            if attempt < max_retries - 1:
                logger.debug(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    raise last_exception


def format_number(num: Union[int, float]) -> str:
    """Format large numbers in human-readable form."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def get_timestamp() -> str:
    """Get UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def safe_format(value: Optional[Union[int, float]], default: str = "N/A",
                precision: int = 0) -> str:
    """Safely format value for display."""
    if value is None:
        return default
    try:
        if isinstance(value, float):
            if precision > 0:
                return f"{value:,.{precision}f}"
            return f"{value:,.0f}"
        else:
            return f"{value:,}"
    except:
        return default


def redact_token(token: Optional[str]) -> str:
    """Redact sensitive token for logging."""
    if not token:
        return "None"
    if len(token) <= 8:
        return "***"
    return token[:4] + "***" + token[-4:]


@contextmanager
def safe_plot(figsize=(10, 12)):
    """Context manager for safe matplotlib plotting."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib not available")
    
    fig = None
    ax = None
    try:
        fig, ax = plt.subplots(figsize=figsize)
        yield fig, ax
    finally:
        if fig is not None:
            plt.close(fig)


@contextmanager
def progress_bar(desc: str, total: Optional[int] = None, disable: bool = False):
    """Context manager for progress bars."""
    if TQDM_AVAILABLE and not disable:
        pbar = tqdm(desc=desc, total=total, unit="step")
        try:
            yield pbar
        finally:
            pbar.close()
    else:
        class DummyProgress:
            def update(self, n=1):
                pass
        yield DummyProgress()


# ============================================================================
# Lazy Pattern Compilation
# ============================================================================

_PATTERNS_NOT_LOADED = object()
_FAMILY_PATTERNS_CACHE = _PATTERNS_NOT_LOADED
_OBJECTIVE_PATTERNS_CACHE = _PATTERNS_NOT_LOADED


def get_family_patterns() -> Dict[ModelFamily, List[re.Pattern]]:
    """Get compiled family patterns (lazy loaded)."""
    global _FAMILY_PATTERNS_CACHE
    
    if _FAMILY_PATTERNS_CACHE is not _PATTERNS_NOT_LOADED:
        return _FAMILY_PATTERNS_CACHE
    
    raw_patterns = {
        ModelFamily.CODELLAMA: [r'codellama', r'code-llama'],
        ModelFamily.LLAMA3: [r'llama-?3', r'llama3'],
        ModelFamily.LLAMA2: [r'llama-?2', r'llama2'],
        ModelFamily.LLAMA: [r'llama(?![23])', r'vicuna', r'alpaca'],
        ModelFamily.MIXTRAL: [r'mixtral', r'mistral.*moe'],
        ModelFamily.MISTRAL: [r'mistral(?!.*moe)', r'zephyr'],
        ModelFamily.QWEN2: [r'qwen-?2', r'qwen2'],
        ModelFamily.QWEN: [r'qwen(?!2)', r'qwen1'],
        ModelFamily.BLOOMZ: [r'bloomz'],
        ModelFamily.BLOOM: [r'bloom(?!z)'],
        ModelFamily.GPT_NEOX: [r'gpt-neox', r'gptneox', r'pythia'],
        ModelFamily.GPT_NEO: [r'gpt-neo', r'gptneo'],
        ModelFamily.GPT_J: [r'gpt-j', r'gptj'],
        ModelFamily.GPT2: [r'gpt2', r'gpt-2', r'distilgpt2'],
        ModelFamily.STARCODER: [r'starcoder', r'starcoderbase'],
        ModelFamily.SANTACODER: [r'santacoder'],
        ModelFamily.CODEGEN: [r'codegen'],
        ModelFamily.OPT: [r'\bopt\b', r'opt-'],
        ModelFamily.FALCON: [r'falcon'],
        ModelFamily.MPT: [r'\bmpt\b', r'mpt-'],
        ModelFamily.STABLELM_EPOCH: [r'stablelm-epoch'],
        ModelFamily.STABLELM: [r'stablelm(?!-epoch)'],
        ModelFamily.REDPAJAMA: [r'redpajama'],
        ModelFamily.BAICHUAN: [r'baichuan'],
        ModelFamily.INTERNLM: [r'internlm'],
        ModelFamily.CHATGLM: [r'chatglm'],
        ModelFamily.PHI3: [r'phi-?3'],
        ModelFamily.PHI2: [r'phi-?2'],
        ModelFamily.PHI: [r'\bphi\b(?![23])', r'phi-1'],
        ModelFamily.GEMMA: [r'gemma'],
        ModelFamily.YI: [r'\byi-\d', r'yi-'],
        ModelFamily.DEEPSEEK: [r'deepseek'],
        ModelFamily.SOLAR: [r'solar'],
        ModelFamily.BERT: [r'\bbert\b', r'bert-'],
        ModelFamily.ROBERTA: [r'roberta'],
        ModelFamily.ALBERT: [r'albert'],
        ModelFamily.ELECTRA: [r'electra'],
        ModelFamily.T5: [r'\bt5\b', r't5-'],
        ModelFamily.BART: [r'\bbart\b', r'bart-'],
        ModelFamily.PEGASUS: [r'pegasus'],
        ModelFamily.CLIP: [r'\bclip\b', r'clip-'],
        ModelFamily.VIT: [r'\bvit\b', r'vision-transformer'],
        ModelFamily.DEIT: [r'deit'],
        ModelFamily.SWIN: [r'swin'],
    }
    
    compiled = {}
    for family, patterns in raw_patterns.items():
        try:
            compiled[family] = [re.compile(p, re.IGNORECASE) for p in patterns]
        except re.error as e:
            logger.warning(f"Invalid regex for {family}: {e}")
            compiled[family] = []
    
    _FAMILY_PATTERNS_CACHE = compiled
    return compiled


def get_objective_patterns() -> Dict[TrainingObjective, List[re.Pattern]]:
    """Get compiled objective patterns (lazy loaded)."""
    global _OBJECTIVE_PATTERNS_CACHE
    
    if _OBJECTIVE_PATTERNS_CACHE is not _PATTERNS_NOT_LOADED:
        return _OBJECTIVE_PATTERNS_CACHE
    
    patterns = {
        TrainingObjective.INSTRUCT: [re.compile(p, re.IGNORECASE) for p in [r'instruct', r'instruction']],
        TrainingObjective.CHAT: [re.compile(p, re.IGNORECASE) for p in [r'chat', r'conversation', r'dialog']],
        TrainingObjective.CODE: [re.compile(p, re.IGNORECASE) for p in [r'code', r'codegen', r'starcoder', r'codellama']],
        TrainingObjective.MATH: [re.compile(p, re.IGNORECASE) for p in [r'math', r'mathstral']],
        TrainingObjective.SUMMARIZATION: [re.compile(p, re.IGNORECASE) for p in [r'summ', r'abstract']],
        TrainingObjective.REWARD: [re.compile(p, re.IGNORECASE) for p in [r'reward', r'rm-']],
    }
    
    _OBJECTIVE_PATTERNS_CACHE = patterns
    return patterns


# ============================================================================
# JSON Encoder
# ============================================================================

class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder with recursion protection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._depth = 0
    
    def encode(self, obj):
        self._depth = 0
        return super().encode(obj)
    
    def default(self, obj):
        self._depth += 1
        
        if self._depth > MAX_JSON_RECURSION_DEPTH:
            return "<max recursion depth exceeded>"
        
        try:
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            return super().default(obj)
        except TypeError:
            return str(obj)
        finally:
            self._depth -= 1


# ============================================================================
# Model Analyzer - COMPLETE WITH ALL FIXES
# ============================================================================

class ModelAnalyzer:
    """Comprehensive model analyzer - COMPLETE WITH ALL FIXES."""
    
    def __init__(self, token: Optional[str] = None, verbose: bool = False, 
                 quiet: bool = False, skip_tokenizer: bool = False):
        """Initialize analyzer."""
        self.token = token or os.environ.get("HF_TOKEN")
        self.verbose = verbose
        self.quiet = quiet
        self.skip_tokenizer = skip_tokenizer
        self.warnings_list: List[str] = []
        
        setup_logging(verbose, quiet)
        
        if self.token:
            logger.debug(f"Using HF token: {redact_token(self.token)}")
        
        try:
            self.api = HfApi(token=self.token)
        except Exception as e:
            logger.error(f"Failed to initialize HF API: {e}")
            self.api = None
    
    def analyze(self, model_id: str) -> Optional[ComprehensiveModelInfo]:
        """Analyze a model comprehensively."""
        is_valid, error_msg = validate_model_id(model_id)
        if not is_valid:
            logger.error(f"Invalid model ID: {error_msg}")
            return None
        
        model_id = model_id.strip()
        self.warnings_list = []
        
        logger.info(f"{'='*80}")
        logger.info(f"🔍 Analyzing Model: {model_id}")
        logger.info(f"{'='*80}")
        
        try:
            with progress_bar("Analyzing model", total=10, disable=self.quiet) as pbar:
                # Load config
                logger.info("📥 Loading configuration...")
                config = self._load_config(model_id)
                if config is None:
                    logger.error("Failed to load model configuration")
                    logger.error("")
                    logger.error("Troubleshooting:")
                    logger.error("  1. Check model ID is correct")
                    logger.error("     Examples: 'gpt2', 'gpt2-xl', 'openai-community/gpt2-xl'")
                    logger.error("  2. Verify internet connection")
                    logger.error("  3. Check model exists on HuggingFace Hub")
                    logger.error("  4. For gated models, provide --token")
                    return None
                pbar.update(1)
                
                # Get metadata
                logger.info("📊 Fetching metadata...")
                metadata = self._get_metadata(model_id)
                pbar.update(1)
                
                # Warn about large models
                if metadata.model_size_mb and metadata.model_size_mb > LARGE_MODEL_WARNING_MB:
                    size_gb = metadata.model_size_mb / MB_TO_GB
                    logger.warning(f"⚠️  Large model: {size_gb:.1f} GB")
                
                # Detect family
                logger.info("🔎 Detecting family...")
                family = self._detect_family(model_id, config)
                pbar.update(1)
                
                # Detect type
                logger.info("🏗️  Determining type...")
                model_type = self._detect_model_type(config)
                pbar.update(1)
                
                # Detect objective
                logger.info("🎯 Identifying objective...")
                objective = self._detect_objective(model_id, config, metadata)
                pbar.update(1)
                
                # Extract architecture
                logger.info("🏛️  Extracting architecture...")
                architecture = self._extract_architecture(config, model_type)
                self._validate_architecture(architecture)
                pbar.update(1)
                
                # Extract attention
                logger.info("👁️  Analyzing attention...")
                attention = self._extract_attention_config(config, architecture)
                pbar.update(1)
                
                # Extract MoE
                logger.info("🎭 Checking MoE...")
                moe = self._extract_moe_config(config)
                if moe.is_moe:
                    logger.info(f"✓ MoE: {moe.num_experts} experts")
                pbar.update(1)
                
                # Load tokenizer
                if not self.skip_tokenizer:
                    logger.info("🔤 Loading tokenizer...")
                    tokenizer_info = self._extract_tokenizer_info(model_id)
                else:
                    logger.info("⏭️  Skipping tokenizer")
                    tokenizer_info = TokenizerInfo()
                pbar.update(1)
                
                # Detect quantization
                logger.info("⚖️  Detecting quantization...")
                quantization = self._detect_quantization(model_id, config)
                
                # Calculate parameters
                logger.info("🔢 Calculating parameters...")
                num_params, trainable_params = self._calculate_parameters(
                    config, architecture, moe, model_type
                )
                if num_params > 0:
                    logger.info(f"✓ Parameters: {format_number(num_params)}")
                else:
                    self.warnings_list.append("Could not calculate parameters")
                
                # Estimate memory
                logger.info("💾 Estimating memory...")
                memory = self._estimate_memory(config, num_params, quantization, 
                                              architecture, attention, moe)
                
                # Determine capabilities
                logger.info("⚡ Determining capabilities...")
                capabilities = self._determine_capabilities(config, architecture, attention)
                pbar.update(1)
            
            # Create info object
            info = ComprehensiveModelInfo(
                model_id=model_id,
                model_family=family,
                model_type=model_type,
                training_objective=objective,
                architecture=architecture,
                attention=attention,
                moe=moe,
                tokenizer=tokenizer_info,
                quantization=quantization,
                num_parameters=num_params,
                num_parameters_human=format_number(num_params),
                trainable_parameters=trainable_params,
                memory=memory,
                capabilities=capabilities,
                metadata=metadata,
                raw_config=safe_config_to_dict(config),
                analysis_timestamp=get_timestamp(),
                warnings=self.warnings_list
            )
            
            logger.info("✅ Analysis complete!")
            
            if self.warnings_list:
                logger.warning(f"\n⚠️  {len(self.warnings_list)} warning(s):")
                for warning in self.warnings_list:
                    logger.warning(f"  - {warning}")
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=self.verbose)
            return None
    
    def _load_config(self, model_id: str) -> Optional[Any]:
        """Load model configuration - FIXED."""
        try:
            # Try direct load first
            try:
                config = AutoConfig.from_pretrained(
                    model_id,
                    token=self.token,
                    trust_remote_code=True
                )
                return config
            except Exception as e:
                logger.debug(f"Direct load failed: {e}")
                
                # Try with retry
                config = retry_with_backoff(
                    AutoConfig.from_pretrained,
                    model_id=model_id,
                    token=self.token,
                    trust_remote_code=True
                )
                return config
                
        except Exception as e:
            logger.debug(f"Config load failed: {e}")
            logger.debug(f"Error type: {type(e).__name__}")
            logger.debug(f"Error details: {str(e)}")
            return None
    
    def _get_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata from HuggingFace Hub."""
        try:
            info = retry_with_backoff(
                model_info, 
                model_id, 
                token=self.token,
                timeout=DEFAULT_TIMEOUT
            )
            
            total_size = 0
            safetensors_size = 0
            if hasattr(info, 'siblings') and info.siblings:
                for sibling in info.siblings:
                    if hasattr(sibling, 'size') and sibling.size:
                        total_size += sibling.size
                        if hasattr(sibling, 'rfilename') and sibling.rfilename.endswith('.safetensors'):
                            safetensors_size += sibling.size
            
            return ModelMetadata(
                model_id=model_id,
                author=getattr(info, 'author', None),
                downloads=getattr(info, 'downloads', 0),
                likes=getattr(info, 'likes', 0),
                tags=list(info.tags) if hasattr(info, 'tags') and info.tags else [],
                library_name=getattr(info, 'library_name', None),
                pipeline_tag=getattr(info, 'pipeline_tag', None),
                license=info.cardData.get('license') if hasattr(info, 'cardData') and info.cardData else None,
                created_at=str(info.created_at) if hasattr(info, 'created_at') else None,
                last_modified=str(info.last_modified) if hasattr(info, 'last_modified') else None,
                model_size_mb=total_size / (1024 * 1024) if total_size > 0 else None,
                safetensors_size_mb=safetensors_size / (1024 * 1024) if safetensors_size > 0 else None,
                is_gated=getattr(info, 'gated', False) if hasattr(info, 'gated') else False,
                gated_reason=info.gated if hasattr(info, 'gated') and isinstance(info.gated, str) else None
            )
        except Exception as e:
            logger.debug(f"Could not fetch metadata: {e}")
            return ModelMetadata(model_id=model_id)
    
    def _detect_family(self, model_id: str, config: Any) -> ModelFamily:
        """Detect model family."""
        model_id_lower = model_id.lower()
        arch_type = safe_getattr(config, 'model_type', default='').lower()
        
        patterns = get_family_patterns()
        
        for family, family_patterns in patterns.items():
            for pattern in family_patterns:
                if pattern.search(model_id_lower) or pattern.search(arch_type):
                    return family
        
        if hasattr(config, 'architectures') and config.architectures:
            arch_str = ' '.join(config.architectures).lower()
            for family, family_patterns in patterns.items():
                for pattern in family_patterns:
                    if pattern.search(arch_str):
                        return family
        
        return ModelFamily.UNKNOWN
    
    def _detect_model_type(self, config: Any) -> ModelType:
        """Detect model type."""
        if safe_getattr(config, 'is_encoder_decoder', default=False):
            return ModelType.ENCODER_DECODER
        
        if hasattr(config, 'architectures') and config.architectures:
            arch_str = ' '.join(config.architectures).lower()
            
            if 'causallm' in arch_str:
                return ModelType.CAUSAL_LM
            elif 'maskedlm' in arch_str:
                return ModelType.MASKED_LM
            elif 'seq2seq' in arch_str or 't5' in arch_str:
                return ModelType.SEQ2SEQ
            elif 'vision' in arch_str or 'vit' in arch_str:
                return ModelType.VISION
            elif 'embedding' in arch_str:
                return ModelType.EMBEDDING
        
        model_type = safe_getattr(config, 'model_type', default='').lower()
        if any(x in model_type for x in ['gpt', 'llama', 'opt', 'mistral', 'falcon', 'bloom']):
            return ModelType.DECODER_ONLY
        elif any(x in model_type for x in ['bert', 'roberta', 'albert']):
            return ModelType.ENCODER_ONLY
        elif any(x in model_type for x in ['t5', 'bart', 'pegasus']):
            return ModelType.ENCODER_DECODER
        elif 'vit' in model_type or 'vision' in model_type:
            return ModelType.VISION
        
        return ModelType.UNKNOWN
    
    def _detect_objective(self, model_id: str, config: Any, 
                         metadata: ModelMetadata) -> TrainingObjective:
        """Detect training objective."""
        model_id_lower = model_id.lower()
        tags_str = ' '.join(metadata.tags).lower()
        
        patterns = get_objective_patterns()
        
        for objective, obj_patterns in patterns.items():
            for pattern in obj_patterns:
                if pattern.search(model_id_lower) or pattern.search(tags_str):
                    return objective
        
        if metadata.pipeline_tag:
            pipeline = metadata.pipeline_tag.lower()
            if 'text-generation' in pipeline:
                return TrainingObjective.BASE
            elif 'summarization' in pipeline:
                return TrainingObjective.SUMMARIZATION
            elif 'translation' in pipeline:
                return TrainingObjective.TRANSLATION
            elif 'question-answering' in pipeline:
                return TrainingObjective.QA
            elif 'text-classification' in pipeline:
                return TrainingObjective.CLASSIFICATION
            elif 'feature-extraction' in pipeline or 'embedding' in pipeline:
                return TrainingObjective.EMBEDDING
        
        return TrainingObjective.BASE
    
    def _extract_architecture(self, config: Any, model_type: ModelType) -> ArchitectureDetails:
        """Extract architecture details - COMPLETE."""
        arch_type = safe_getattr(config, 'model_type', default='unknown')
        
        architectures = safe_getattr(config, 'architectures', default=[])
        if architectures:
            arch_type = architectures[0]
        
        # Extract and validate dimensions
        hidden_size = validate_positive_int(
            safe_getattr(config, 'hidden_size', 'n_embd', 'd_model'),
            'hidden_size',
            min_val=MIN_REASONABLE_HIDDEN,
            max_val=MAX_REASONABLE_HIDDEN
        )
        
        num_hidden_layers = validate_positive_int(
            safe_getattr(config, 'num_hidden_layers', 'n_layer', 'num_layers'),
            'num_hidden_layers',
            max_val=MAX_REASONABLE_LAYERS
        )
        
        vocab_size = validate_positive_int(
            safe_getattr(config, 'vocab_size'),
            'vocab_size',
            max_val=MAX_REASONABLE_VOCAB
        )
        
        num_attention_heads = validate_positive_int(
            safe_getattr(config, 'num_attention_heads', 'n_head'),
            'num_attention_heads'
        )
        
        intermediate_size = validate_positive_int(
            safe_getattr(config, 'intermediate_size', 'n_inner', 'ffn_dim'),
            'intermediate_size'
        )
        
        max_pos_emb = validate_positive_int(
            safe_getattr(config, 'max_position_embeddings', 'n_positions', 'max_sequence_length'),
            'max_position_embeddings',
            max_val=MAX_REASONABLE_CONTEXT
        )
        
        # Vision-specific
        image_size = validate_positive_int(
            safe_getattr(config, 'image_size'),
            'image_size'
        )
        
        patch_size = validate_positive_int(
            safe_getattr(config, 'patch_size'),
            'patch_size'
        )
        
        num_channels = validate_positive_int(
            safe_getattr(config, 'num_channels'),
            'num_channels'
        )
        
        # Dropout validation
        attention_dropout = validate_float_range(
            safe_getattr(config, 'attention_dropout', 'attn_pdrop', 'attention_probs_dropout_prob'),
            'attention_dropout'
        )
        
        hidden_dropout = validate_float_range(
            safe_getattr(config, 'hidden_dropout', 'resid_pdrop', 'hidden_dropout_prob'),
            'hidden_dropout'
        )
        
        residual_dropout = validate_float_range(
            safe_getattr(config, 'residual_dropout'),
            'residual_dropout'
        )
        
        details = ArchitectureDetails(
            architecture_type=arch_type,
            num_layers=validate_positive_int(safe_getattr(config, 'num_layers'), 'num_layers'),
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=validate_positive_int(
                safe_getattr(config, 'num_key_value_heads'),
                'num_key_value_heads'
            ),
            max_position_embeddings=max_pos_emb,
            vocab_size=vocab_size,
            
            encoder_layers=validate_positive_int(
                safe_getattr(config, 'encoder_layers', 'num_encoder_layers'),
                'encoder_layers'
            ),
            decoder_layers=validate_positive_int(
                safe_getattr(config, 'decoder_layers', 'num_decoder_layers'),
                'decoder_layers'
            ),
            encoder_hidden_size=validate_positive_int(
                safe_getattr(config, 'encoder_hidden_size', 'd_model'),
                'encoder_hidden_size'
            ),
            decoder_hidden_size=validate_positive_int(
                safe_getattr(config, 'decoder_hidden_size', 'd_model'),
                'decoder_hidden_size'
            ),
            
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            
            use_cache=safe_getattr(config, 'use_cache', default=True),
            tie_word_embeddings=safe_getattr(config, 'tie_word_embeddings', default=False),
            is_encoder_decoder=safe_getattr(config, 'is_encoder_decoder', default=False),
            layer_norm_type=safe_getattr(config, 'layer_norm_type'),
            layer_norm_eps=safe_getattr(config, 'layer_norm_eps', 'layer_norm_epsilon'),
            rms_norm_eps=safe_getattr(config, 'rms_norm_eps'),
            activation_function=safe_getattr(config, 'activation_function'),
            hidden_act=safe_getattr(config, 'hidden_act'),
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            residual_dropout=residual_dropout,
            use_bias=safe_getattr(config, 'use_bias', default=True),
            attention_bias=safe_getattr(config, 'attention_bias', default=True)
        )
        
        config_dict = safe_config_to_dict(config)
        known_keys = set(details.__dict__.keys())
        details.additional_config = {
            k: v for k, v in config_dict.items() 
            if k not in known_keys and not k.startswith('_')
        }
        
        return details
    
    def _validate_architecture(self, arch: ArchitectureDetails):
        """Validate architecture consistency."""
        if arch.hidden_size and arch.num_attention_heads:
            if arch.num_attention_heads > 0 and arch.hidden_size % arch.num_attention_heads != 0:
                self.warnings_list.append(
                    f"hidden_size ({arch.hidden_size}) not divisible by num_attention_heads ({arch.num_attention_heads})"
                )
        
        if arch.num_key_value_heads and arch.num_attention_heads:
            if arch.num_key_value_heads > arch.num_attention_heads:
                self.warnings_list.append(
                    f"num_key_value_heads ({arch.num_key_value_heads}) > num_attention_heads ({arch.num_attention_heads})"
                )
        
        if arch.intermediate_size and arch.hidden_size:
            ratio = arch.intermediate_size / arch.hidden_size
            if ratio < 1 or ratio > 16:
                self.warnings_list.append(
                    f"Unusual intermediate_size ratio: {ratio:.1f}x hidden_size"
                )
    
    def _extract_attention_config(self, config: Any, arch: ArchitectureDetails) -> AttentionConfig:
        """Extract attention configuration."""
        num_heads = arch.num_attention_heads
        num_kv_heads = arch.num_key_value_heads or num_heads
        hidden_size = arch.hidden_size
        
        attention_type = AttentionType.STANDARD
        
        sliding_window = validate_positive_int(
            safe_getattr(config, 'sliding_window', 'sliding_window_size'),
            'sliding_window'
        )
        
        if num_kv_heads is not None and num_heads is not None and num_kv_heads != num_heads:
            if num_kv_heads == 1:
                attention_type = AttentionType.MULTI_QUERY
            elif num_kv_heads < num_heads:
                attention_type = AttentionType.GROUPED_QUERY
        
        if sliding_window is not None and attention_type == AttentionType.STANDARD:
            attention_type = AttentionType.SLIDING_WINDOW
        
        head_dim = None
        if hidden_size is not None and num_heads is not None and num_heads > 0:
            head_dim = hidden_size // num_heads
            head_dim = validate_positive_int(head_dim, 'head_dim')
        
        use_flash = (
            safe_getattr(config, 'use_flash_attention', default=False) or
            safe_getattr(config, '_flash_attn_2_enabled', default=False) or
            safe_getattr(config, 'use_flash_attention_2', default=False)
        )
        
        return AttentionConfig(
            attention_type=attention_type,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window_size=sliding_window,
            use_flash_attention=use_flash,
            use_alibi=safe_getattr(config, 'alibi', default=False),
            use_rope=safe_getattr(config, 'rope', default=False),
            rope_theta=safe_getattr(config, 'rope_theta'),
            rope_scaling=safe_getattr(config, 'rope_scaling')
        )
    
    def _extract_moe_config(self, config: Any) -> MoEConfig:
        """Extract MoE configuration."""
        num_experts = validate_positive_int(
            safe_getattr(
                config, 
                'num_local_experts', 
                'num_experts',
                'moe_num_experts',
                'n_routed_experts'
            ),
            'num_experts'
        )
        
        if num_experts is None:
            return MoEConfig(is_moe=False)
        
        return MoEConfig(
            is_moe=True,
            num_experts=num_experts,
            num_experts_per_token=validate_positive_int(
                safe_getattr(
                    config, 
                    'num_experts_per_tok', 
                    'num_experts_per_token',
                    'moe_top_k'
                ),
                'num_experts_per_token'
            ),
            expert_capacity=validate_positive_int(
                safe_getattr(config, 'expert_capacity'),
                'expert_capacity'
            ),
            router_type=safe_getattr(config, 'router_type', 'routing_type'),
            expert_hidden_size=validate_positive_int(
                safe_getattr(config, 'expert_hidden_size'),
                'expert_hidden_size'
            ),
            shared_expert=safe_getattr(config, 'shared_expert', default=False),
            num_shared_experts=validate_positive_int(
                safe_getattr(config, 'num_shared_experts'),
                'num_shared_experts'
            )
        )
    
    def _extract_tokenizer_info(self, model_id: str) -> TokenizerInfo:
        """Extract tokenizer information - COMPLETELY FIXED."""
        try:
            # Load tokenizer (direct load, no retry)
            logger.debug(f"Loading tokenizer for {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=self.token,
                trust_remote_code=True
            )
            
            logger.debug(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
            
            # Safe attribute getter with error handling
            def safe_get(attr: str, convert_to_str: bool = False) -> Optional[Any]:
                try:
                    val = getattr(tokenizer, attr, None)
                    if val is None:
                        return None
                    if convert_to_str:
                        return str(val)
                    return val
                except Exception as e:
                    logger.debug(f"Could not get {attr}: {e}")
                    return None
            
            # Vocab size
            vocab_size = None
            try:
                if hasattr(tokenizer, '__len__'):
                    vocab_size = len(tokenizer)
                else:
                    vocab_size = safe_get('vocab_size')
                
                if vocab_size and vocab_size > MAX_REASONABLE_VOCAB:
                    logger.debug(f"Vocab size {vocab_size} exceeds maximum, setting to None")
                    vocab_size = None
                elif vocab_size:
                    vocab_size = int(vocab_size)
            except Exception as e:
                logger.debug(f"Could not get vocab size: {e}")
            
            # Model max length
            max_len = safe_get('model_max_length')
            if max_len and max_len > MAX_REASONABLE_CONTEXT:
                logger.debug(f"model_max_length {max_len} exceeds maximum, setting to None")
                max_len = None
            elif max_len:
                try:
                    max_len = int(max_len)
                except:
                    max_len = None
            
            # Chat template
            chat_template = None
            has_chat_template = False
            try:
                if hasattr(tokenizer, 'chat_template'):
                    chat_template = tokenizer.chat_template
                    if chat_template is not None:
                        chat_template = str(chat_template)
                        has_chat_template = True
            except Exception as e:
                logger.debug(f"Could not get chat template: {e}")
            
            # Additional special tokens
            additional_special_tokens = []
            try:
                tokens = safe_get('additional_special_tokens')
                if tokens:
                    additional_special_tokens = [str(t) for t in tokens]
            except Exception as e:
                logger.debug(f"Could not get additional_special_tokens: {e}")
            
            # Token IDs with validation
            def safe_get_id(attr: str) -> Optional[int]:
                try:
                    val = safe_get(attr)
                    if val is None:
                        return None
                    val = int(val)
                    if val < 0:
                        return None
                    return val
                except:
                    return None
            
            info = TokenizerInfo(
                tokenizer_type=tokenizer.__class__.__name__,
                vocab_size=vocab_size,
                model_max_length=max_len,
                bos_token=safe_get('bos_token', True),
                eos_token=safe_get('eos_token', True),
                unk_token=safe_get('unk_token', True),
                sep_token=safe_get('sep_token', True),
                pad_token=safe_get('pad_token', True),
                cls_token=safe_get('cls_token', True),
                mask_token=safe_get('mask_token', True),
                bos_token_id=safe_get_id('bos_token_id'),
                eos_token_id=safe_get_id('eos_token_id'),
                unk_token_id=safe_get_id('unk_token_id'),
                sep_token_id=safe_get_id('sep_token_id'),
                pad_token_id=safe_get_id('pad_token_id'),
                additional_special_tokens=additional_special_tokens,
                do_lower_case=safe_get('do_lower_case') or False,
                add_prefix_space=safe_get('add_prefix_space') or False,
                chat_template=chat_template,
                has_chat_template=has_chat_template
            )
            
            logger.debug(f"✓ Tokenizer info extracted successfully")
            return info
            
        except Exception as e:
            logger.debug(f"Tokenizer loading failed: {type(e).__name__}: {e}")
            if self.verbose:
                import traceback
                logger.debug(traceback.format_exc())
            self.warnings_list.append(f"Tokenizer loading failed: {str(e)}")
            return TokenizerInfo()
    
    def _detect_quantization(self, model_id: str, config: Any) -> QuantizationInfo:
        """Detect quantization - COMPLETE."""
        model_id_lower = model_id.lower()
        
        quantization_config = safe_getattr(config, 'quantization_config')
        
        quant_type = QuantizationType.NONE
        bits = 32.0
        
        if quantization_config is not None:
            quant_method = safe_getattr(quantization_config, 'quant_method', default='').lower()
            
            if 'gptq' in quant_method:
                quant_type = QuantizationType.GPTQ
                bits = float(safe_getattr(quantization_config, 'bits', default=4))
            elif 'awq' in quant_method:
                quant_type = QuantizationType.AWQ
                bits = float(safe_getattr(quantization_config, 'bits', default=4))
            elif 'bitsandbytes' in quant_method or 'bnb' in quant_method:
                load_in_4bit = safe_getattr(quantization_config, 'load_in_4bit', default=False)
                load_in_8bit = safe_getattr(quantization_config, 'load_in_8bit', default=False)
                if load_in_4bit:
                    quant_type = QuantizationType.BNB_4BIT
                    bits = 4.0
                elif load_in_8bit:
                    quant_type = QuantizationType.BNB_8BIT
                    bits = 8.0
        
        if quant_type == QuantizationType.NONE:
            quant_patterns = [
                (r'gptq', QuantizationType.GPTQ, 4.0),
                (r'awq', QuantizationType.AWQ, 4.0),
                (r'q4[-_]k[-_]m', QuantizationType.GGUF_Q4_K_M, 4.0),
                (r'q5[-_]k[-_]s', QuantizationType.GGUF_Q5_K_S, 5.0),
                (r'q6[-_]k', QuantizationType.GGUF_Q6_K, 6.0),
                (r'q8[-_]0', QuantizationType.GGUF_Q8_0, 8.0),
                (r'gguf', QuantizationType.GGUF, 4.0),
                (r'ggml', QuantizationType.GGML, 4.0),
                (r'exl2', QuantizationType.EXL2, 4.0),
                (r'int8|8bit', QuantizationType.INT8, 8.0),
                (r'int4|4bit', QuantizationType.INT4, 4.0),
                (r'int2|2bit', QuantizationType.INT2, 2.0),
            ]
            
            for pattern, qtype, qbits in quant_patterns:
                if re.search(pattern, model_id_lower, re.IGNORECASE):
                    quant_type = qtype
                    bits = qbits
                    break
        
        if bits not in [2, 3, 4, 5, 6, 8, 16, 32]:
            self.warnings_list.append(f"Unusual quantization bits: {bits}")
        
        info = QuantizationInfo(
            quantization_type=quant_type,
            bits_per_param=bits,
            is_quantized=quant_type != QuantizationType.NONE,
            quantization_config=safe_config_to_dict(quantization_config) if quantization_config else None
        )
        
        if quantization_config is not None:
            if quant_type == QuantizationType.GPTQ:
                info.gptq_bits = validate_positive_int(
                    safe_getattr(quantization_config, 'bits'),
                    'gptq_bits'
                )
                info.gptq_group_size = validate_positive_int(
                    safe_getattr(quantization_config, 'group_size'),
                    'gptq_group_size'
                )
                info.gptq_desc_act = safe_getattr(quantization_config, 'desc_act', default=False)
            
            elif quant_type == QuantizationType.AWQ:
                info.awq_bits = validate_positive_int(
                    safe_getattr(quantization_config, 'bits'),
                    'awq_bits'
                )
                info.awq_group_size = validate_positive_int(
                    safe_getattr(quantization_config, 'group_size'),
                    'awq_group_size'
                )
                info.awq_version = safe_getattr(quantization_config, 'version')
            
            elif quant_type in [QuantizationType.BNB_4BIT, QuantizationType.BNB_8BIT]:
                info.bnb_4bit_compute_dtype = safe_getattr(quantization_config, 'bnb_4bit_compute_dtype')
                info.bnb_4bit_quant_type = safe_getattr(quantization_config, 'bnb_4bit_quant_type')
                info.bnb_4bit_use_double_quant = safe_getattr(
                    quantization_config, 'bnb_4bit_use_double_quant', default=False
                )
        
        return info
    
    def _calculate_parameters(self, config: Any, architecture: ArchitectureDetails, 
                             moe: MoEConfig, model_type: ModelType) -> Tuple[int, int]:
        """Calculate parameters - COMPLETE."""
        if hasattr(config, 'num_parameters'):
            num_params = validate_positive_int(
                config.num_parameters, 
                'num_parameters',
                max_val=MAX_REASONABLE_PARAMS
            )
            if num_params is not None:
                return num_params, num_params
        
        if model_type == ModelType.ENCODER_DECODER:
            return self._calculate_encoder_decoder_params(architecture, moe)
        elif model_type == ModelType.VISION:
            return self._calculate_vision_params(architecture)
        else:
            return self._calculate_decoder_params(architecture, moe)
    
    def _calculate_decoder_params(self, arch: ArchitectureDetails, 
                                  moe: MoEConfig) -> Tuple[int, int]:
        """Calculate decoder parameters - COMPLETE."""
        hidden_size = arch.hidden_size
        num_layers = arch.num_hidden_layers or arch.num_layers
        intermediate_size = arch.intermediate_size
        vocab_size = arch.vocab_size
        num_heads = arch.num_attention_heads
        num_kv_heads = arch.num_key_value_heads or num_heads
        
        if not all([hidden_size, num_layers, vocab_size]):
            return 0, 0
        
        if num_heads is None or num_heads == 0:
            num_heads = 32
            self.warnings_list.append("num_attention_heads missing, using 32")
        if num_kv_heads is None or num_kv_heads == 0:
            num_kv_heads = num_heads
        
        try:
            # Embedding
            embedding_params = vocab_size * hidden_size
            
            # Attention per layer
            head_dim = hidden_size // num_heads
            
            if num_kv_heads != num_heads:
                # GQA/MQA
                q_params = hidden_size * hidden_size
                kv_params = 2 * hidden_size * (num_kv_heads * head_dim)
                o_params = hidden_size * hidden_size
                attn_params_per_layer = q_params + kv_params + o_params
                
                if arch.attention_bias:
                    attn_params_per_layer += hidden_size  # Q bias
                    attn_params_per_layer += (num_kv_heads * head_dim)  # K bias
                    attn_params_per_layer += (num_kv_heads * head_dim)  # V bias
                    attn_params_per_layer += hidden_size  # O bias
            else:
                # Standard attention
                attn_params_per_layer = 4 * hidden_size * hidden_size
                if arch.attention_bias:
                    attn_params_per_layer += 4 * hidden_size
            
            # FFN per layer
            if intermediate_size:
                ffn_params_per_layer = 2 * hidden_size * intermediate_size
                if arch.use_bias:
                    ffn_params_per_layer += intermediate_size  # Up projection bias
                    ffn_params_per_layer += hidden_size  # Down projection bias
            else:
                ffn_params_per_layer = 2 * hidden_size * (4 * hidden_size)
                if arch.use_bias:
                    ffn_params_per_layer += (4 * hidden_size) + hidden_size
            
            # Layer norm (2 params: gamma and beta)
            ln_params_per_layer = 2 * hidden_size * 2  # 2 norms per layer
            
            # Total per layer
            params_per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
            
            # MoE adjustment
            if moe.is_moe and moe.num_experts:
                params_per_layer -= ffn_params_per_layer
                
                if moe.num_shared_experts:
                    routed_experts = moe.num_experts - moe.num_shared_experts
                    params_per_layer += ffn_params_per_layer * routed_experts
                    params_per_layer += ffn_params_per_layer * moe.num_shared_experts
                else:
                    params_per_layer += ffn_params_per_layer * moe.num_experts
                
                # Router
                router_params = hidden_size * moe.num_experts
                if arch.use_bias:
                    router_params += moe.num_experts
                params_per_layer += router_params
            
            # Total
            total_params = embedding_params + (num_layers * params_per_layer)
            
            # Final layer norm
            total_params += 2 * hidden_size
            
            # LM head (don't add if tied)
            if not arch.tie_word_embeddings:
                total_params += vocab_size * hidden_size
                if arch.use_bias:
                    total_params += vocab_size
            
            if total_params < 0 or total_params > MAX_REASONABLE_PARAMS:
                self.warnings_list.append(f"Unusual parameter count: {format_number(total_params)}")
                return 0, 0
            
            return int(total_params), int(total_params)
            
        except Exception as e:
            logger.debug(f"Error calculating parameters: {e}")
            return 0, 0
    
    def _calculate_encoder_decoder_params(self, arch: ArchitectureDetails,
                                         moe: MoEConfig) -> Tuple[int, int]:
        """Calculate encoder-decoder parameters - COMPLETE."""
        encoder_hidden = arch.encoder_hidden_size or arch.hidden_size
        decoder_hidden = arch.decoder_hidden_size or arch.hidden_size
        encoder_layers = arch.encoder_layers or (arch.num_hidden_layers or 0) // 2
        decoder_layers = arch.decoder_layers or (arch.num_hidden_layers or 0) // 2
        
        if not all([encoder_hidden, decoder_hidden, encoder_layers, decoder_layers, arch.vocab_size]):
            return 0, 0
        
        try:
            # Embeddings
            embedding_params = arch.vocab_size * encoder_hidden
            
            # Encoder
            encoder_params_per_layer = 4 * encoder_hidden * encoder_hidden  # Self-attn
            encoder_params_per_layer += 2 * encoder_hidden * (4 * encoder_hidden)  # FFN
            encoder_params_per_layer += 4 * encoder_hidden  # Layer norms
            if arch.use_bias:
                encoder_params_per_layer += 4 * encoder_hidden  # Attention bias
                encoder_params_per_layer += (4 * encoder_hidden) + encoder_hidden  # FFN bias
            
            # Decoder
            decoder_params_per_layer = 4 * decoder_hidden * decoder_hidden  # Self-attn
            # Cross-attention
            decoder_params_per_layer += decoder_hidden * decoder_hidden  # Q
            decoder_params_per_layer += 2 * decoder_hidden * encoder_hidden  # K, V
            decoder_params_per_layer += decoder_hidden * decoder_hidden  # O
            decoder_params_per_layer += 2 * decoder_hidden * (4 * decoder_hidden)  # FFN
            decoder_params_per_layer += 6 * decoder_hidden  # Layer norms
            if arch.use_bias:
                decoder_params_per_layer += 4 * decoder_hidden  # Self-attn bias
                decoder_params_per_layer += 4 * decoder_hidden  # Cross-attn bias
                decoder_params_per_layer += (4 * decoder_hidden) + decoder_hidden  # FFN bias
            
            total_params = (
                embedding_params +
                encoder_layers * encoder_params_per_layer +
                decoder_layers * decoder_params_per_layer +
                arch.vocab_size * decoder_hidden
            )
            
            if arch.use_bias:
                total_params += arch.vocab_size
            
            if total_params < 0 or total_params > MAX_REASONABLE_PARAMS:
                return 0, 0
            
            return int(total_params), int(total_params)
            
        except Exception as e:
            logger.debug(f"Error calculating encoder-decoder parameters: {e}")
            return 0, 0
    
    def _calculate_vision_params(self, arch: ArchitectureDetails) -> Tuple[int, int]:
        """Calculate vision model parameters - COMPLETE."""
        if not all([arch.hidden_size, arch.num_hidden_layers, arch.image_size, arch.patch_size]):
            return 0, 0
        
        try:
            # Number of patches
            num_patches = (arch.image_size // arch.patch_size) ** 2
            
            # Patch embedding
            patch_dim = (arch.num_channels or 3) * (arch.patch_size ** 2)
            patch_embed_params = patch_dim * arch.hidden_size
            if arch.use_bias:
                patch_embed_params += arch.hidden_size
            
            # Position embedding
            pos_embed_params = (num_patches + 1) * arch.hidden_size  # +1 for CLS
            
            # CLS token
            cls_token_params = arch.hidden_size
            
            # Transformer blocks
            params_per_layer = 4 * arch.hidden_size * arch.hidden_size  # Attention
            params_per_layer += 2 * arch.hidden_size * (4 * arch.hidden_size)  # FFN
            params_per_layer += 4 * arch.hidden_size  # Layer norms
            
            if arch.use_bias:
                params_per_layer += 4 * arch.hidden_size  # Attention bias
                params_per_layer += (4 * arch.hidden_size) + arch.hidden_size  # FFN bias
            
            # Total
            total_params = (
                patch_embed_params +
                pos_embed_params +
                cls_token_params +
                arch.num_hidden_layers * params_per_layer +
                2 * arch.hidden_size  # Final layer norm
            )
            
            # Classification head
            if arch.vocab_size:
                total_params += arch.hidden_size * arch.vocab_size
                if arch.use_bias:
                    total_params += arch.vocab_size
            
            return int(total_params), int(total_params)
            
        except Exception as e:
            logger.debug(f"Error calculating vision parameters: {e}")
            return 0, 0
    
    def _estimate_memory(self, config: Any, num_params: int, 
                        quantization: QuantizationInfo,
                        architecture: ArchitectureDetails,
                        attention: AttentionConfig,
                        moe: MoEConfig) -> MemoryEstimate:
        """Estimate memory - COMPLETE."""
        if num_params <= 0:
            return MemoryEstimate()
        
        try:
            # Parameter memory
            params_fp32 = num_params * BYTES_PER_PARAM['fp32'] / (1024 * 1024)
            params_fp16 = num_params * BYTES_PER_PARAM['fp16'] / (1024 * 1024)
            params_int8 = num_params * BYTES_PER_PARAM['int8'] / (1024 * 1024)
            params_int4 = num_params * BYTES_PER_PARAM['int4'] / (1024 * 1024)
            
            # Activation memory
            hidden_size = architecture.hidden_size or 4096
            num_layers = architecture.num_hidden_layers or architecture.num_layers or 32
            seq_len = DEFAULT_SEQ_LENGTH
            
            activation_mb = (2 * hidden_size * seq_len * num_layers * BYTES_PER_PARAM['fp16']) / (1024 * 1024)
            
            # MoE activation adjustment
            if moe.is_moe and moe.num_experts_per_token:
                moe_multiplier = 1 + (moe.num_experts_per_token / (moe.num_experts or 1))
                activation_mb *= moe_multiplier
            
            # KV cache
            num_heads = architecture.num_attention_heads or 32
            num_kv_heads = architecture.num_key_value_heads or num_heads
            
            if num_heads == 0:
                num_heads = 32
            if num_kv_heads == 0:
                num_kv_heads = num_heads
            
            head_dim = attention.head_dim or (hidden_size // num_heads)
            
            # Use num_kv_heads for cache
            kv_per_token = (2 * num_layers * num_kv_heads * head_dim * BYTES_PER_PARAM['fp16']) / (1024 * 1024)
            
            max_pos = architecture.max_position_embeddings or DEFAULT_SEQ_LENGTH
            kv_full = kv_per_token * max_pos
            
            # Total inference
            total_inf_fp16 = params_fp16 + activation_mb + kv_full
            total_inf_int8 = params_int8 + activation_mb + kv_full
            
            # Training memory
            total_train_fp32 = params_fp32 * 4 + activation_mb * 2
            total_train_fp32_adam = params_fp32 * (1 + OPTIMIZER_MEMORY_MULTIPLIER['adam'] * 2) + activation_mb * 2
            
            return MemoryEstimate(
                params_fp32_mb=params_fp32,
                params_fp16_mb=params_fp16,
                params_int8_mb=params_int8,
                params_int4_mb=params_int4,
                activation_memory_mb=activation_mb,
                kv_cache_per_token_mb=kv_per_token,
                kv_cache_full_context_mb=kv_full,
                total_inference_fp16_mb=total_inf_fp16,
                total_inference_int8_mb=total_inf_int8,
                total_training_fp32_mb=total_train_fp32,
                total_training_fp32_adam_mb=total_train_fp32_adam
            )
        except Exception as e:
            logger.debug(f"Error estimating memory: {e}")
            return MemoryEstimate()
    
    def _determine_capabilities(self, config: Any, architecture: ArchitectureDetails, 
                                attention: AttentionConfig) -> ModelCapabilities:
        """Determine capabilities."""
        is_causal = False
        if hasattr(config, 'architectures') and config.architectures:
            arch_str = ' '.join(config.architectures).lower()
            is_causal = 'causallm' in arch_str or 'lmhead' in arch_str
        
        max_context = architecture.max_position_embeddings
        long_context = max_context is not None and max_context > 8192
        
        supports_gc = safe_getattr(config, 'gradient_checkpointing', default=False)
        
        return ModelCapabilities(
            can_generate=is_causal,
            supports_beam_search=is_causal,
            supports_sampling=is_causal,
            max_context_length=max_context,
            supports_long_context=long_context,
            supports_streaming=is_causal,
            supports_batching=True,
            supports_gradient_checkpointing=supports_gc,
            is_trainable=True,
            supports_lora=True,
            supports_qlora=True,
            supports_flash_attention=attention.use_flash_attention,
            supports_torch_compile=True,
            supports_bettertransformer=not attention.use_flash_attention
        )


# ============================================================================
# Visualization - COMPLETE
# ============================================================================

class ModelVisualizer:
    """Visualize model architecture - COMPLETE."""
    
    def __init__(self, info: ComprehensiveModelInfo):
        self.info = info
    
    def visualize_architecture(self, output_file: str = "model_architecture.png", 
                               style: str = "detailed"):
        """Create visualization."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available - install with: pip install matplotlib")
            return
        
        is_valid, error_msg = validate_output_path(output_file)
        if not is_valid:
            logger.error(f"Invalid output path: {error_msg}")
            return
        
        try:
            if style == "simple":
                self._visualize_simple(output_file)
            elif style == "detailed":
                self._visualize_detailed(output_file)
            else:
                self._visualize_detailed(output_file)
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
    
    def _visualize_simple(self, output_file: str):
        """Simple visualization - COMPLETE."""
        with safe_plot(figsize=(10, 12)) as (fig, ax):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 12)
            ax.axis('off')
            
            ax.text(5, 11.5, f"{self.info.model_id}", 
                    ha='center', va='top', fontsize=16, fontweight='bold')
            ax.text(5, 11, f"{self.info.num_parameters_human} parameters", 
                    ha='center', va='top', fontsize=12, color='gray')
            
            y = 10
            
            self._draw_box(ax, 2, y, 6, 0.5, "Input Tokens", "lightblue")
            y -= 1
            
            vocab_size = safe_format(self.info.architecture.vocab_size)
            self._draw_box(ax, 2, y, 6, 0.5, 
                          f"Token Embedding\n({vocab_size} vocab)", 
                          "lightgreen")
            y -= 1
            
            num_layers = self.info.architecture.num_hidden_layers or self.info.architecture.num_layers or 1
            self._draw_box(ax, 2, y, 6, 0.8, 
                          f"Transformer Layers × {num_layers}", 
                          "lightyellow")
            
            layer_y = y - 0.1
            self._draw_box(ax, 2.5, layer_y - 0.2, 5, 0.15, "Self-Attention", "wheat", fontsize=8)
            self._draw_box(ax, 2.5, layer_y - 0.4, 5, 0.15, "Feed-Forward", "wheat", fontsize=8)
            
            y -= 1.5
            
            self._draw_box(ax, 2, y, 6, 0.5, 
                          f"Output Head\n({vocab_size} vocab)", 
                          "lightcoral")
            y -= 1
            
            self._draw_box(ax, 2, y, 6, 0.5, "Output Tokens", "lightblue")
            
            # Info panel
            info_y = 5
            info_text = [
                f"Family: {self.info.model_family.value}",
                f"Type: {self.info.model_type.value}",
                f"Hidden: {safe_format(self.info.architecture.hidden_size)}",
                f"Heads: {safe_format(self.info.architecture.num_attention_heads)}",
                f"Context: {safe_format(self.info.capabilities.max_context_length)}",
            ]
            
            for i, text in enumerate(info_text):
                ax.text(0.5, info_y - i * 0.3, text, fontsize=9, family='monospace')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved visualization to {output_file}")
    
    def _visualize_detailed(self, output_file: str):
        """Detailed visualization - COMPLETE."""
        with safe_plot(figsize=(14, 16)) as (fig, ax):
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 16)
            ax.axis('off')
            
            ax.text(7, 15.5, f"{self.info.model_id}", 
                    ha='center', va='top', fontsize=18, fontweight='bold')
            ax.text(7, 15, f"{self.info.model_family.value.upper()} | {self.info.num_parameters_human} params", 
                    ha='center', va='top', fontsize=12, color='gray')
            
            y = 14
            
            self._draw_box(ax, 4, y, 6, 0.5, "Input Token IDs", "lightblue")
            self._draw_arrow(ax, 7, y - 0.5, 7, y - 0.8)
            y -= 1.3
            
            vocab = safe_format(self.info.architecture.vocab_size)
            hidden = safe_format(self.info.architecture.hidden_size)
            self._draw_box(ax, 4, y, 6, 0.6, 
                          f"Token Embedding\nVocab: {vocab}\nDim: {hidden}", 
                          "lightgreen")
            self._draw_arrow(ax, 7, y - 0.6, 7, y - 0.9)
            y -= 1.5
            
            if self.info.architecture.max_position_embeddings:
                max_pos = safe_format(self.info.architecture.max_position_embeddings)
                self._draw_box(ax, 4, y, 6, 0.5, 
                              f"Position Embedding\nMax Pos: {max_pos}", 
                              "lightgreen")
                self._draw_arrow(ax, 7, y - 0.5, 7, y - 0.8)
                y -= 1.3
            
            num_layers = self.info.architecture.num_hidden_layers or self.info.architecture.num_layers or 1
            
            block_height = 3.5
            self._draw_box(ax, 3, y - block_height, 8, block_height, 
                          f"Transformer Block × {num_layers}", 
                          "lightyellow", alpha=0.3)
            
            layer_y = y - 0.5
            
            attn_label = "Multi-Head Self-Attention"
            if self.info.attention.attention_type == AttentionType.GROUPED_QUERY:
                attn_label = "Grouped-Query Attention"
            elif self.info.attention.attention_type == AttentionType.MULTI_QUERY:
                attn_label = "Multi-Query Attention"
            
            num_heads = safe_format(self.info.architecture.num_attention_heads)
            attn_info = f"{attn_label}\nHeads: {num_heads}"
            if self.info.architecture.num_key_value_heads:
                kv_heads = safe_format(self.info.architecture.num_key_value_heads)
                attn_info += f" | KV: {kv_heads}"
            
            self._draw_box(ax, 3.5, layer_y, 7, 0.8, attn_info, "wheat")
            self._draw_arrow(ax, 7, layer_y - 0.8, 7, layer_y - 1.1)
            layer_y -= 1.4
            
            norm_type = "RMSNorm" if self.info.architecture.rms_norm_eps else "LayerNorm"
            self._draw_box(ax, 4.5, layer_y, 5, 0.4, norm_type, "lavender")
            self._draw_arrow(ax, 7, layer_y - 0.4, 7, layer_y - 0.7)
            layer_y -= 1.0
            
            ffn_info = f"Feed-Forward Network\nHidden: {hidden}"
            if self.info.architecture.intermediate_size:
                inter = safe_format(self.info.architecture.intermediate_size)
                ffn_info += f"\nIntermediate: {inter}"
            
            self._draw_box(ax, 3.5, layer_y, 7, 0.8, ffn_info, "lightcyan")
            self._draw_arrow(ax, 7, layer_y - 0.8, 7, layer_y - 1.1)
            layer_y -= 1.4
            
            self._draw_box(ax, 4.5, layer_y, 5, 0.4, norm_type, "lavender")
            
            y = layer_y - 0.9
            self._draw_arrow(ax, 7, y, 7, y - 0.3)
            y -= 0.8
            
            self._draw_box(ax, 4.5, y, 5, 0.4, f"Final {norm_type}", "lavender")
            self._draw_arrow(ax, 7, y - 0.4, 7, y - 0.7)
            y -= 1.1
            
            self._draw_box(ax, 4, y, 6, 0.6, 
                          f"Language Model Head\nOutput: {vocab} logits", 
                          "lightcoral")
            self._draw_arrow(ax, 7, y - 0.6, 7, y - 0.9)
            y -= 1.5
            
            self._draw_box(ax, 4, y, 6, 0.5, "Output Probabilities", "lightblue")
            
            # Info panel
            info_x = 11
            info_y = 14
            
            info_items = [
                ("Architecture", self.info.architecture.architecture_type),
                ("Family", self.info.model_family.value),
                ("Type", self.info.model_type.value),
                ("", ""),
                ("Parameters", self.info.num_parameters_human),
                ("Hidden Size", hidden),
                ("Layers", str(num_layers)),
                ("Heads", num_heads),
                ("", ""),
                ("Max Context", safe_format(self.info.capabilities.max_context_length)),
                ("Vocab Size", vocab),
                ("", ""),
                ("Memory (FP16)", f"{self.info.memory.params_fp16_mb / MB_TO_GB:.1f} GB"),
                ("Memory (INT8)", f"{self.info.memory.params_int8_mb / MB_TO_GB:.1f} GB"),
            ]
            
            if self.info.moe.is_moe:
                info_items.extend([
                    ("", ""),
                    ("MoE Experts", str(self.info.moe.num_experts)),
                    ("Active", str(self.info.moe.num_experts_per_token)),
                ])
            
            for i, (key, value) in enumerate(info_items):
                if key == "":
                    continue
                ax.text(info_x, info_y - i * 0.35, f"{key}:", fontsize=9, fontweight='bold')
                ax.text(info_x + 2, info_y - i * 0.35, value, fontsize=9, family='monospace')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved detailed visualization to {output_file}")
    
    def _draw_box(self, ax, x, y, width, height, text, color, alpha=1.0, fontsize=10):
        """Draw box with text."""
        rect = FancyBboxPatch((x, y - height), width, height,
                             boxstyle="round,pad=0.05", 
                             facecolor=color, edgecolor='black', 
                             linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + width/2, y - height/2, text, 
               ha='center', va='center', fontsize=fontsize, wrap=True)
    
    def _draw_arrow(self, ax, x1, y1, x2, y2):
        """Draw arrow."""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)


# ============================================================================
# Output Formatters - COMPLETE
# ============================================================================

class OutputFormatter:
    """Format model information - COMPLETE."""
    
    @staticmethod
    def print_summary(info: ComprehensiveModelInfo):
        """Print summary."""
        print(f"\n{'='*80}")
        print(f"📊 MODEL SUMMARY: {info.model_id}")
        print(f"{'='*80}\n")
        
        print(f"Family:              {info.model_family.value}")
        print(f"Type:                {info.model_type.value}")
        print(f"Objective:           {info.training_objective.value}")
        print(f"Parameters:          {info.num_parameters_human} ({info.num_parameters:,})")
        print(f"Architecture:        {info.architecture.architecture_type}")
        
        print(f"\nDimensions:")
        print(f"  Hidden Size:       {safe_format(info.architecture.hidden_size)}")
        print(f"  Layers:            {safe_format(info.architecture.num_hidden_layers or info.architecture.num_layers)}")
        print(f"  Attention Heads:   {safe_format(info.architecture.num_attention_heads)}")
        print(f"  Vocab Size:        {safe_format(info.architecture.vocab_size)}")
        print(f"  Max Context:       {safe_format(info.capabilities.max_context_length)}")
        
        print(f"\nMemory (Parameters):")
        print(f"  FP32:              {info.memory.params_fp32_mb / MB_TO_GB:.2f} GB")
        print(f"  FP16:              {info.memory.params_fp16_mb / MB_TO_GB:.2f} GB")
        print(f"  INT8:              {info.memory.params_int8_mb / MB_TO_GB:.2f} GB")
        print(f"  INT4:              {info.memory.params_int4_mb / MB_TO_GB:.2f} GB")
        
        if info.tokenizer.tokenizer_type:
            print(f"\nTokenizer:")
            print(f"  Type:              {info.tokenizer.tokenizer_type}")
            print(f"  Vocab Size:        {safe_format(info.tokenizer.vocab_size)}")
            print(f"  Chat Template:     {'Yes' if info.tokenizer.has_chat_template else 'No'}")
        
        if info.quantization.is_quantized:
            print(f"\nQuantization:")
            print(f"  Type:              {info.quantization.quantization_type.value}")
            print(f"  Bits:              {info.quantization.bits_per_param}")
        
        if info.moe.is_moe:
            print(f"\nMixture of Experts:")
            print(f"  Num Experts:       {info.moe.num_experts}")
            print(f"  Active per Token:  {info.moe.num_experts_per_token}")
        
        print(f"\n{'='*80}\n")
    
    @staticmethod
    def print_detailed(info: ComprehensiveModelInfo):
        """Print detailed information - COMPLETE."""
        print(f"\n{'='*80}")
        print(f"📊 DETAILED MODEL ANALYSIS: {info.model_id}")
        print(f"{'='*80}\n")
        
        # Basic Info
        print("BASIC INFORMATION")
        print("=" * 80)
        print(f"Model ID:            {info.model_id}")
        print(f"Family:              {info.model_family.value}")
        print(f"Type:                {info.model_type.value}")
        print(f"Training Objective:  {info.training_objective.value}")
        print(f"Architecture:        {info.architecture.architecture_type}")
        print(f"Parameters:          {info.num_parameters_human} ({info.num_parameters:,})")
        print(f"Trainable Params:    {info.trainable_parameters:,}")
        
        # Architecture
        print(f"\n{'='*80}")
        print("ARCHITECTURE DETAILS")
        print("=" * 80)
        print(f"Num Layers:          {safe_format(info.architecture.num_hidden_layers or info.architecture.num_layers)}")
        print(f"Hidden Size:         {safe_format(info.architecture.hidden_size)}")
        print(f"Intermediate Size:   {safe_format(info.architecture.intermediate_size)}")
        print(f"Attention Heads:     {safe_format(info.architecture.num_attention_heads)}")
        if info.architecture.num_key_value_heads:
            print(f"KV Heads:            {safe_format(info.architecture.num_key_value_heads)}")
        print(f"Max Position Emb:    {safe_format(info.architecture.max_position_embeddings)}")
        print(f"Vocab Size:          {safe_format(info.architecture.vocab_size)}")
        print(f"Activation:          {info.architecture.hidden_act or info.architecture.activation_function or 'N/A'}")
        print(f"Use Cache:           {info.architecture.use_cache}")
        print(f"Tie Embeddings:      {info.architecture.tie_word_embeddings}")
        
        # Attention
        print(f"\n{'='*80}")
        print("ATTENTION MECHANISM")
        print("=" * 80)
        print(f"Type:                {info.attention.attention_type.value}")
        print(f"Num Heads:           {safe_format(info.attention.num_attention_heads)}")
        if info.attention.num_key_value_heads:
            print(f"KV Heads:            {safe_format(info.attention.num_key_value_heads)}")
        if info.attention.head_dim:
            print(f"Head Dimension:      {safe_format(info.attention.head_dim)}")
        print(f"Flash Attention:     {info.attention.use_flash_attention}")
        print(f"RoPE:                {info.attention.use_rope}")
        
        # MoE
        if info.moe.is_moe:
            print(f"\n{'='*80}")
            print("MIXTURE OF EXPERTS")
            print("=" * 80)
            print(f"Num Experts:         {info.moe.num_experts}")
            print(f"Experts per Token:   {info.moe.num_experts_per_token}")
            if info.moe.num_shared_experts:
                print(f"Shared Experts:      {info.moe.num_shared_experts}")
        
        # Tokenizer
        if info.tokenizer.tokenizer_type:
            print(f"\n{'='*80}")
            print("TOKENIZER")
            print("=" * 80)
            print(f"Type:                {info.tokenizer.tokenizer_type}")
            print(f"Vocab Size:          {safe_format(info.tokenizer.vocab_size)}")
            print(f"Model Max Length:    {safe_format(info.tokenizer.model_max_length)}")
            print(f"BOS Token:           {info.tokenizer.bos_token or 'N/A'}")
            print(f"EOS Token:           {info.tokenizer.eos_token or 'N/A'}")
            print(f"PAD Token:           {info.tokenizer.pad_token or 'N/A'}")
            print(f"Chat Template:       {'Yes' if info.tokenizer.has_chat_template else 'No'}")
        
        # Quantization
        if info.quantization.is_quantized:
            print(f"\n{'='*80}")
            print("QUANTIZATION")
            print("=" * 80)
            print(f"Type:                {info.quantization.quantization_type.value}")
            print(f"Bits per Param:      {info.quantization.bits_per_param}")
        
        # Memory
        print(f"\n{'='*80}")
        print("MEMORY ESTIMATES")
        print("=" * 80)
        print(f"Parameters:")
        print(f"  FP32:              {info.memory.params_fp32_mb / MB_TO_GB:.2f} GB")
        print(f"  FP16:              {info.memory.params_fp16_mb / MB_TO_GB:.2f} GB")
        print(f"  INT8:              {info.memory.params_int8_mb / MB_TO_GB:.2f} GB")
        print(f"  INT4:              {info.memory.params_int4_mb / MB_TO_GB:.2f} GB")
        print(f"\nKV Cache:")
        print(f"  Per Token:         {info.memory.kv_cache_per_token_mb:.2f} MB")
        print(f"  Full Context:      {info.memory.kv_cache_full_context_mb / MB_TO_GB:.2f} GB")
        print(f"\nTotal Estimates:")
        print(f"  Inference (FP16):  {info.memory.total_inference_fp16_mb / MB_TO_GB:.2f} GB")
        print(f"  Training (FP32):   {info.memory.total_training_fp32_mb / MB_TO_GB:.2f} GB")
        print(f"  Training (Adam):   {info.memory.total_training_fp32_adam_mb / MB_TO_GB:.2f} GB")
        
        # Capabilities
        print(f"\n{'='*80}")
        print("CAPABILITIES")
        print("=" * 80)
        print(f"Can Generate:        {info.capabilities.can_generate}")
        print(f"Max Context:         {safe_format(info.capabilities.max_context_length)}")
        print(f"Long Context:        {info.capabilities.supports_long_context}")
        print(f"Flash Attention:     {info.capabilities.supports_flash_attention}")
        print(f"LoRA:                {info.capabilities.supports_lora}")
        print(f"QLoRA:               {info.capabilities.supports_qlora}")
        
        # Metadata
        print(f"\n{'='*80}")
        print("METADATA")
        print("=" * 80)
        print(f"Author:              {info.metadata.author or 'Unknown'}")
        print(f"Downloads:           {info.metadata.downloads:,}")
        print(f"Likes:               {info.metadata.likes:,}")
        print(f"License:             {info.metadata.license or 'Unknown'}")
        if info.metadata.model_size_mb:
            print(f"Model Size:          {info.metadata.model_size_mb / MB_TO_GB:.2f} GB")
        
        print(f"\n{'='*80}\n")
    
    @staticmethod
    def export_json(info: ComprehensiveModelInfo, output_file: str):
        """Export to JSON - COMPLETE."""
        is_valid, error_msg = validate_output_path(output_file)
        if not is_valid:
            logger.error(f"Invalid output path: {error_msg}")
            return
        
        def to_dict(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):
                        result[key] = to_dict(value)
                return result
            elif isinstance(obj, (list, tuple)):
                return [to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        try:
            data = to_dict(info)
            
            json_str = json.dumps(data, indent=2, cls=EnhancedJSONEncoder)
            size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
            
            if size_mb > MAX_JSON_SIZE_MB:
                logger.warning(f"JSON export is large: {size_mb:.1f} MB")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            logger.info(f"✅ Exported to {output_file} ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}", exc_info=True)
    
    @staticmethod
    def export_markdown(info: ComprehensiveModelInfo, output_file: str):
        """Export to Markdown - COMPLETE."""
        is_valid, error_msg = validate_output_path(output_file)
        if not is_valid:
            logger.error(f"Invalid output path: {error_msg}")
            return
        
        try:
            lines = []
            lines.append(f"# Model Analysis: {info.model_id}\n")
            lines.append(f"**Analysis Date:** {info.analysis_timestamp}\n")
            
            lines.append("## Basic Information\n")
            lines.append(f"- **Family:** {info.model_family.value}")
            lines.append(f"- **Type:** {info.model_type.value}")
            lines.append(f"- **Parameters:** {info.num_parameters_human} ({info.num_parameters:,})")
            lines.append(f"- **Architecture:** {info.architecture.architecture_type}\n")
            
            lines.append("## Architecture\n")
            lines.append(f"- **Layers:** {safe_format(info.architecture.num_hidden_layers or info.architecture.num_layers)}")
            lines.append(f"- **Hidden Size:** {safe_format(info.architecture.hidden_size)}")
            lines.append(f"- **Attention Heads:** {safe_format(info.architecture.num_attention_heads)}")
            lines.append(f"- **Vocab Size:** {safe_format(info.architecture.vocab_size)}\n")
            
            lines.append("## Memory Requirements\n")
            lines.append("| Precision | Size |")
            lines.append("|-----------|------|")
            lines.append(f"| FP32 | {info.memory.params_fp32_mb / MB_TO_GB:.2f} GB |")
            lines.append(f"| FP16 | {info.memory.params_fp16_mb / MB_TO_GB:.2f} GB |")
            lines.append(f"| INT8 | {info.memory.params_int8_mb / MB_TO_GB:.2f} GB |")
            lines.append(f"| INT4 | {info.memory.params_int4_mb / MB_TO_GB:.2f} GB |\n")
            
            if info.tokenizer.tokenizer_type:
                lines.append("## Tokenizer\n")
                lines.append(f"- **Type:** {info.tokenizer.tokenizer_type}")
                lines.append(f"- **Vocab Size:** {safe_format(info.tokenizer.vocab_size)}")
                lines.append(f"- **Chat Template:** {'Yes' if info.tokenizer.has_chat_template else 'No'}\n")
            
            if info.moe.is_moe:
                lines.append("## Mixture of Experts\n")
                lines.append(f"- **Num Experts:** {info.moe.num_experts}")
                lines.append(f"- **Active per Token:** {info.moe.num_experts_per_token}\n")
            
            content = '\n'.join(lines)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Exported to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export Markdown: {e}", exc_info=True)


# ============================================================================
# CLI - COMPLETE
# ============================================================================

def main():
    """Main CLI entry point - COMPLETE."""
    parser = argparse.ArgumentParser(
        description=f"ModelAnalyzer v{__version__} - COMPLETE FIXED VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python modelanalyzer.py gpt2
  python modelanalyzer.py gpt2-xl --detailed
  python modelanalyzer.py openai-community/gpt2-xl --visualize
  python modelanalyzer.py facebook/opt-6.7b --detailed --visualize
  python modelanalyzer.py meta-llama/Llama-2-7b-hf --export analysis.json --token hf_xxxxx
  python modelanalyzer.py mistralai/Mistral-7B-v0.1 --quiet --export results.json
        """
    )
    
    parser.add_argument("model_id", type=str, nargs='?', help="HuggingFace model ID")
    
    # Output options
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    
    # Export options
    parser.add_argument("--export", type=str, help="Export to JSON file")
    parser.add_argument("--export-markdown", type=str, help="Export to Markdown file")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Create architecture visualization")
    parser.add_argument("--viz-output", type=str, default="model_architecture.png", help="Visualization output file")
    parser.add_argument("--viz-style", type=str, default="detailed", choices=["simple", "detailed"], help="Visualization style")
    
    # Other options
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer loading (faster)")
    parser.add_argument("--token", type=str, help="HuggingFace API token")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    
    args = parser.parse_args()
    
    # Version
    if args.version:
        print(f"ModelAnalyzer v{__version__}")
        return
    
    # Require model_id
    if not args.model_id:
        parser.print_help()
        return
    
    # Create analyzer
    analyzer = ModelAnalyzer(
        token=args.token,
        verbose=args.verbose,
        quiet=args.quiet,
        skip_tokenizer=args.skip_tokenizer
    )
    
    # Analyze model
    if not args.quiet:
        print(f"\n🔍 Analyzing {args.model_id}...")
    
    info = analyzer.analyze(args.model_id)
    
    if not info:
        sys.exit(1)
    
    # Display results
    if not args.quiet:
        if args.detailed:
            OutputFormatter.print_detailed(info)
        else:
            OutputFormatter.print_summary(info)
    
    # Export
    if args.export:
        OutputFormatter.export_json(info, args.export)
    
    if args.export_markdown:
        OutputFormatter.export_markdown(info, args.export_markdown)
    
    # Visualize
    if args.visualize:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("⚠️  Visualization skipped - matplotlib not available")
            logger.warning("   Install with: pip install matplotlib")
        else:
            if not args.quiet:
                print(f"\n🎨 Creating {args.viz_style} visualization...")
            visualizer = ModelVisualizer(info)
            visualizer.visualize_architecture(args.viz_output, args.viz_style)
    
    if not args.quiet:
        print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
