# Model Analyzer - a tool to analyze LLM models (Transformer architecture) on HuggingFace


- ModelAnalyzer v2.0.2 - User Manual
- Author: Michael Stal, 2026
- Version: 2.0.2
- License: MIT

## Table of Contents

- 1.  Introduction
- 2.  What is ModelAnalyzer?
- 3.  How It Works
- 4.  Installation
- 5.  Basic Usage
- 6.  HuggingFace Token Configuration
- 7.  Command-Line Parameters
- 8.  Output and Results
- 9.  Advanced Usage Examples
- 10. Troubleshooting
- 11. Technical Details
- 12. FAQ
- 13. Appendix


## 1. Introduction

ModelAnalyzer is a comprehensive command-line tool for analyzing transformer-based language models from the HuggingFace Hub. It provides detailed insights into model architecture, memory requirements, capabilities, and more—without downloading the full model weights.

### Key Features

- 🔍 Deep Architecture Analysis - Extract detailed model configuration
- 💾 Memory Estimation - Calculate memory requirements for inference and training
- 📊 Parameter Counting - Accurate parameter calculation for all architectures
- 🎨 Visualization - Generate architecture diagrams
- 📤 Export - Save analysis results as JSON or Markdown
- 🔤 Tokenizer Analysis - Examine tokenizer configuration
- ⚖️ Quantization Detection - Identify quantization methods
- 🎭 MoE Support - Analyze Mixture-of-Experts models
- 🚀 40+ Model Families - Support for GPT, LLaMA, Mistral, BERT, T5, and more


## 2. What is ModelAnalyzer?

### Purpose

ModelAnalyzer helps researchers, engineers, and ML practitioners understand transformer models before deployment. It answers critical questions:

- How many parameters does this model have?
- How much memory will it require?
- What architecture does it use?
- Does it support long context?
- Is it quantized?
- What tokenizer does it use?

### Use Cases

- Model Selection - Compare models before downloading
- Resource Planning - Estimate GPU/CPU requirements
- Research - Study architecture patterns across model families
- Documentation - Generate technical specifications
- Debugging - Verify model configurations
- Education - Learn about transformer architectures

### What It Does NOT Do

- ❌ Download full model weights
- ❌ Run inference or generate text
- ❌ Fine-tune or train models
- ❌ Modify model files
- ❌ Require GPU


## 3. How It Works

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    ModelAnalyzer v2.0.2                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   HuggingFace Hub API                       │
│  • Model metadata (downloads, likes, license)               │
│  • Config files (config.json)                               │
│  • Tokenizer files (tokenizer.json, tokenizer_config.json)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Analysis Pipeline                      │
│                                                             │
│  1. Config Loading      → Load model configuration          │
│  2. Metadata Fetching   → Get model info from Hub          │
│  3. Family Detection    → Identify model family            │
│  4. Type Detection      → Determine model type             │
│  5. Architecture Extract → Parse architecture details       │
│  6. Attention Analysis  → Analyze attention mechanism       │
│  7. MoE Detection       → Check for Mixture-of-Experts      │
│  8. Tokenizer Loading   → Extract tokenizer info           │
│  9. Quantization Check  → Detect quantization               │
│  10. Parameter Count    → Calculate total parameters        │
│  11. Memory Estimation  → Estimate memory requirements      │
│  12. Capability Check   → Determine model capabilities      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Output Generation                      │
│                                                             │
│  • Console Output (summary/detailed)                        │
│  • JSON Export                                              │
│  • Markdown Export                                          │
│  • Architecture Visualization (PNG)                         │
└─────────────────────────────────────────────────────────────┘

### Analysis Process

- Validation - Validates model ID format
- Config Loading - Downloads config.json from HuggingFace Hub
- Pattern Matching - Uses regex patterns to detect model family
- Architecture Parsing - Extracts dimensions, layers, heads, etc.
- Parameter Calculation - Computes total parameters based on architecture
- Memory Estimation - Calculates memory for different precisions
- Output Formatting - Presents results in requested format
```

### Supported Model Families



Family
Examples
Detection



GPT-2
gpt2, gpt2-xl, distilgpt2
Model ID, architecture


GPT-J
EleutherAI/gpt-j-6b
Model ID, architecture


GPT-NeoX
EleutherAI/pythia-*
Model ID, architecture


LLaMA
meta-llama/Llama-2-7b-hf
Model ID, architecture


LLaMA 2
meta-llama/Llama-2-13b-hf
Model ID, architecture


LLaMA 3
meta-llama/Meta-Llama-3-8B
Model ID, architecture


Mistral
mistralai/Mistral-7B-v0.1
Model ID, architecture


Mixtral
mistralai/Mixtral-8x7B-v0.1
Model ID, MoE config


Qwen
Qwen/Qwen-7B
Model ID, architecture


Qwen2
Qwen/Qwen2-7B
Model ID, architecture


Phi
microsoft/phi-1, phi-2, phi-3
Model ID, architecture


Gemma
google/gemma-7b
Model ID, architecture


Falcon
tiiuae/falcon-7b
Model ID, architecture


MPT
mosaicml/mpt-7b
Model ID, architecture


BLOOM
bigscience/bloom-7b1
Model ID, architecture


OPT
facebook/opt-6.7b
Model ID, architecture


BERT
bert-base-uncased
Model ID, architecture


RoBERTa
roberta-base
Model ID, architecture


T5
t5-base, t5-large
Model ID, architecture


BART
facebook/bart-base
Model ID, architecture


ViT
google/vit-base-patch16-224
Model ID, architecture


And 20+ more...





## 4. Installation

### Prerequisites

Python: 3.8 or higher
Operating System: Linux, macOS, or Windows
Internet Connection: Required for accessing HuggingFace Hub

Step 1: Install Python Dependencies

Option A: Using pip (Recommended)
#Install required packages
pip install transformers huggingface-hub

#Optional: Install visualization support
pip install matplotlib

#Optional: Install progress bars
pip install tqdm

Option B: Using conda
#Create conda environment
conda create -n modelanalyzer python=3.10
conda activate modelanalyzer

#Install dependencies
conda install -c conda-forge transformers huggingface-hub matplotlib tqdm

Option C: Using requirements.txt
Create requirements.txt:
transformers>=4.30.0
huggingface-hub>=0.16.0
matplotlib>=3.5.0
tqdm>=4.65.0

Install:
pip install -r requirements.txt

Step 2: Download ModelAnalyzer
Save the modelanalyzer.py script to your working directory.
Step 3: Verify Installation

#Check Python version
python --version

#Verify dependencies
python -c "import transformers; print(transformers.__version__)"
python -c "import huggingface_hub; print(huggingface_hub.__version__)"

#Test ModelAnalyzer
python modelanalyzer.py --version

Expected output:
ModelAnalyzer v2.0.2

Troubleshooting Installation
Problem: ModuleNotFoundError: No module named 'transformers'
Solution:
pip install --upgrade transformers huggingface-hub

Problem: ImportError: cannot import name 'AutoConfig'
Solution:
pip install --upgrade transformers

Problem: Visualization doesn't work
Solution:
pip install matplotlib


## 5. Basic Usage

### Quickstart

#Analyze a model (simplest form)
python modelanalyzer.py gpt2

#Analyze with detailed output
python modelanalyzer.py gpt2 --detailed

#Analyze with visualization
python modelanalyzer.py gpt2 --visualize

Basic Syntax
python modelanalyzer.py <model_id> [OPTIONS]

### Model ID Formats

ModelAnalyzer accepts various model ID formats:

#Short form (official models)
python modelanalyzer.py gpt2
python modelanalyzer.py bert-base-uncased

#Full form (organization/model)
python modelanalyzer.py openai-community/gpt2-xl
python modelanalyzer.py meta-llama/Llama-2-7b-hf
python modelanalyzer.py mistralai/Mistral-7B-v0.1

#Model variants
python modelanalyzer.py gpt2-medium
python modelanalyzer.py gpt2-large
python modelanalyzer.py gpt2-xl

Example: Analyzing GPT-2 XL
python modelanalyzer.py gpt2-xl

Output:
```
================================================================================
🔍 Analyzing Model: gpt2-xl
================================================================================
📥 Loading configuration...
📊 Fetching metadata...
🔎 Detecting family...
🏗️  Determining type...
🎯 Identifying objective...
🏛️  Extracting architecture...
👁️  Analyzing attention...
🎭 Checking MoE...
🔤 Loading tokenizer...
⚖️  Detecting quantization...
🔢 Calculating parameters...
✓ Parameters: 1.56B
💾 Estimating memory...
⚡ Determining capabilities...
✅ Analysis complete!

================================================================================
📊 MODEL SUMMARY: gpt2-xl
================================================================================

Family:              gpt2
Type:                causal-lm
Objective:           base
Parameters:          1.56B (1,557,611,200)
Architecture:        GPT2LMHeadModel

Dimensions:
  Hidden Size:       1,600
  Layers:            48
  Attention Heads:   25
  Vocab Size:        50,257
  Max Context:       1,024

Memory (Parameters):
  FP32:              5.78 GB
  FP16:              2.89 GB
  INT8:              1.45 GB
  INT4:              0.72 GB

Tokenizer:
  Type:              GPT2TokenizerFast
  Vocab Size:        50,257
  Chat Template:     No

================================================================================
```

## 6. HuggingFace Token Configuration

### When Do You Need a Token?
A HuggingFace token is required for:

#### Gated Models - Models requiring license acceptance

- LLaMA 2, LLaMA 3
- Some Mistral models
- Gemma models
- Other restricted models


#### Private Models - Your own private repositories

#### Rate Limiting - Higher API rate limits with authentication


### When You DON'T Need a Token

- Public models (GPT-2, BERT, T5, etc.)
- Most open-source models
- Community models without restrictions

### How to Get a Token

Create HuggingFace Account

Go to https://huggingface.co/join
Sign up with email


### Generate Token

Visit https://huggingface.co/settings/tokens
Click "New token"
Name: ModelAnalyzer
Type: Read
Click "Generate token"
Copy the token (starts with hf_)


### Accept Model License (for gated models)

Visit model page (e.g., https://huggingface.co/meta-llama/Llama-2-7b-hf)
Click "Agree and access repository"
Fill out form and submit



### Token Configuration Methods

Method 1: Command-Line Argument (Recommended for One-Time Use)
python modelanalyzer.py meta-llama/Llama-2-7b-hf --token hf_YourTokenHere

Pros:

- Simple and direct
- No configuration needed

Cons:

- Token visible in command history
- Must type token each time

Method 2: Environment Variable (Recommended for Regular Use)
Linux/macOS:

#Temporary (current session)
export HF_TOKEN="hf_YourTokenHere"
python modelanalyzer.py meta-llama/Llama-2-7b-hf

#Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export HF_TOKEN="hf_YourTokenHere"' >> ~/.bashrc
source ~/.bashrc

Windows (Command Prompt):
#Temporary
set HF_TOKEN=hf_YourTokenHere
python modelanalyzer.py meta-llama/Llama-2-7b-hf

#Permanent (System Environment Variables)
setx HF_TOKEN "hf_YourTokenHere"

Windows (PowerShell):
#Temporary
$env:HF_TOKEN="hf_YourTokenHere"
python modelanalyzer.py meta-llama/Llama-2-7b-hf

#Permanent (User profile)
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_YourTokenHere", "User")

Pros:

- Token not visible in command
- Works for all commands
- Secure

Cons:

- Requires shell configuration

Method 3: HuggingFace CLI Login (Recommended for Best Security)

#Install HuggingFace CLI (if not installed)
pip install huggingface-hub

#Login (token stored securely)
huggingface-cli login

#Enter token when prompted
#Token is saved to ~/.cache/huggingface/token

#Now use ModelAnalyzer without specifying token
python modelanalyzer.py meta-llama/Llama-2-7b-hf

Pros:

- Most secure method
- Token stored encrypted
- Works across all HuggingFace tools
- No need to specify token each time

Cons:

- Requires HuggingFace CLI installation

### Token Security Best Practices
✅ DO:

- Use environment variables or HuggingFace CLI
- Keep tokens private
- Regenerate tokens periodically
- Use read-only tokens

❌ DON'T:

- Commit tokens to Git repositories
- Share tokens publicly
- Use write tokens for read-only tasks
- Hardcode tokens in scripts

Example: Analyzing Gated Model


#Step 1: Accept license at https://huggingface.co/meta-llama/Llama-2-7b-hf

#Step 2: Login with HuggingFace CLI
huggingface-cli login

#Step 3: Analyze model
python modelanalyzer.py meta-llama/Llama-2-7b-hf --detailed --visualize

#Alternative: Use token directly
python modelanalyzer.py meta-llama/Llama-2-7b-hf \
    --token hf_YourTokenHere \
    --detailed \
    --visualize


## 7. Command-Line Parameters

### Complete Reference
python modelanalyzer.py <model_id> [OPTIONS]

#### Positional Arguments



Argument
Description
Required



model_id
HuggingFace model identifier
Yes


#### Output Options



Option
Description
Default



--detailed
Show detailed analysis instead of summary
False


--verbose
Enable verbose logging (debug mode)
False


--quiet
Suppress all non-error output
False


Examples:
#Summary output (default)
python modelanalyzer.py gpt2

#Detailed output
python modelanalyzer.py gpt2 --detailed

#Verbose mode (for debugging)
python modelanalyzer.py gpt2 --verbose

#Quiet mode (errors only)
python modelanalyzer.py gpt2 --quiet

#### Export Options



Option
Description
Default



--export <file>
Export analysis to JSON file
None


--export-markdown <file>
Export analysis to Markdown file
None


Examples:
# Export to JSON
python modelanalyzer.py gpt2 --export gpt2_analysis.json

# Export to Markdown
python modelanalyzer.py gpt2 --export-markdown gpt2_analysis.md

# Export to both
python modelanalyzer.py gpt2 \
    --export gpt2.json \
    --export-markdown gpt2.md

#### Visualization Options



Option
Description
Default



--visualize
Create architecture visualization
False


--viz-output <file>
Visualization output file
model_architecture.png


--viz-style <style>
Visualization style: simple or detailed
detailed


Examples:
#Create visualization (default detailed style)
python modelanalyzer.py gpt2 --visualize

#Simple visualization
python modelanalyzer.py gpt2 --visualize --viz-style simple

#Custom output file
python modelanalyzer.py gpt2 \
    --visualize \
    --viz-output gpt2_architecture.png

#Detailed visualization with custom name
python modelanalyzer.py gpt2 \
    --visualize \
    --viz-style detailed \
    --viz-output gpt2_detailed.png
    

#### Performance Options



Option
Description
Default



--skip-tokenizer
Skip tokenizer loading (faster analysis)
False


Examples:

#Skip tokenizer for faster analysis
python modelanalyzer.py gpt2 --skip-tokenizer

#Useful for batch processing
for model in gpt2 gpt2-medium gpt2-large; do
    python modelanalyzer.py $model --skip-tokenizer --quiet --export ${model}.json
done

#### Authentication Options



Option
Description
Default



--token <token>
HuggingFace API token
$HF_TOKEN env var


Examples:
#Use token from command line
python modelanalyzer.py meta-llama/Llama-2-7b-hf --token hf_YourToken

#Use token from environment variable (recommended)
export HF_TOKEN="hf_YourToken"
python modelanalyzer.py meta-llama/Llama-2-7b-hf

#### Utility Options



Option
Description



--version
Show version and exit


--help, -h
Show help message and exit


Examples:
#Show version
python modelanalyzer.py --version

#Show help
python modelanalyzer.py --help

Option Combinations
Complete Analysis with All Outputs
python modelanalyzer.py gpt2-xl \
    --detailed \
    --visualize \
    --viz-style detailed \
    --export gpt2xl_analysis.json \
    --export-markdown gpt2xl_analysis.md \
    --viz-output gpt2xl_architecture.png

Fast Batch Analysis
python modelanalyzer.py gpt2 \
    --quiet \
    --skip-tokenizer \
    --export gpt2.json

Debug Mode
python modelanalyzer.py problematic-model \
    --verbose \
    --detailed

Gated Model Analysis
python modelanalyzer.py meta-llama/Llama-2-7b-hf \
    --token hf_YourToken \
    --detailed \
    --visualize \
    --export llama2_analysis.json


8. Output and Results
Console Output Formats
Summary Output (Default)
python modelanalyzer.py gpt2-xl

Output Structure:
```
================================================================================
📊 MODEL SUMMARY: gpt2-xl
================================================================================

Family:              gpt2
Type:                causal-lm
Objective:           base
Parameters:          1.56B (1,557,611,200)
Architecture:        GPT2LMHeadModel

Dimensions:
  Hidden Size:       1,600
  Layers:            48
  Attention Heads:   25
  Vocab Size:        50,257
  Max Context:       1,024

Memory (Parameters):
  FP32:              5.78 GB
  FP16:              2.89 GB
  INT8:              1.45 GB
  INT4:              0.72 GB

Tokenizer:
  Type:              GPT2TokenizerFast
  Vocab Size:        50,257
  Chat Template:     No

================================================================================
```

### Detailed Output
python modelanalyzer.py gpt2-xl --detailed

Output Structure:
```
================================================================================
📊 DETAILED MODEL ANALYSIS: gpt2-xl
================================================================================

BASIC INFORMATION
================================================================================
Model ID:            gpt2-xl
Family:              gpt2
Type:                causal-lm
Training Objective:  base
Architecture:        GPT2LMHeadModel
Parameters:          1.56B (1,557,611,200)
Trainable Params:    1,557,611,200

================================================================================
ARCHITECTURE DETAILS
================================================================================
Num Layers:          48
Hidden Size:         1,600
Intermediate Size:   6,400
Attention Heads:     25
Max Position Emb:    1,024
Vocab Size:          50,257
Activation:          gelu_new
Use Cache:           True
Tie Embeddings:      True

================================================================================
ATTENTION MECHANISM
================================================================================
Type:                standard
Num Heads:           25
Head Dimension:      64
Flash Attention:     False
RoPE:                False

================================================================================
TOKENIZER
================================================================================
Type:                GPT2TokenizerFast
Vocab Size:          50,257
Model Max Length:    1,024
BOS Token:           <|endoftext|>
EOS Token:           <|endoftext|>
PAD Token:           <|endoftext|>
Chat Template:       No

================================================================================
MEMORY ESTIMATES
================================================================================
Parameters:
  FP32:              5.78 GB
  FP16:              2.89 GB
  INT8:              1.45 GB
  INT4:              0.72 GB

KV Cache:
  Per Token:         0.30 MB
  Full Context:      307.20 MB

Total Estimates:
  Inference (FP16):  3.49 GB
  Training (FP32):   25.31 GB
  Training (Adam):   40.27 GB

================================================================================
CAPABILITIES
================================================================================
Can Generate:        True
Max Context:         1,024
Long Context:        False
Flash Attention:     False
LoRA:                True
QLoRA:               True

================================================================================
METADATA
================================================================================
Author:              openai
Downloads:           2,345,678
Likes:               1,234
License:             mit
Model Size:          6.03 GB

================================================================================
```

#### JSON Export

python modelanalyzer.py gpt2 --export gpt2_analysis.json

File Structure:
```
{
  "model_id": "gpt2",
  "model_family": "gpt2",
  "model_type": "causal-lm",
  "training_objective": "base",
  "architecture": {
    "architecture_type": "GPT2LMHeadModel",
    "num_hidden_layers": 12,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "max_position_embeddings": 1024,
    "vocab_size": 50257,
    "use_cache": true,
    "tie_word_embeddings": true,
    "activation_function": "gelu_new"
  },
  "attention": {
    "attention_type": "standard",
    "num_attention_heads": 12,
    "head_dim": 64,
    "use_flash_attention": false
  },
  "tokenizer": {
    "tokenizer_type": "GPT2TokenizerFast",
    "vocab_size": 50257,
    "model_max_length": 1024,
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "has_chat_template": false
  },
  "num_parameters": 124439808,
  "num_parameters_human": "124.44M",
  "memory": {
    "params_fp32_mb": 474.49,
    "params_fp16_mb": 237.24,
    "params_int8_mb": 118.62,
    "params_int4_mb": 59.31,
    "kv_cache_per_token_mb": 0.023,
    "total_inference_fp16_mb": 337.24,
    "total_training_fp32_mb": 2072.45
  },
  "capabilities": {
    "can_generate": true,
    "max_context_length": 1024,
    "supports_long_context": false,
    "supports_lora": true
  },
  "metadata": {
    "author": "openai",
    "downloads": 12345678,
    "likes": 5432,
    "license": "mit"
  },
  "analysis_timestamp": "2026-02-20T08:33:15.123456Z"
}
```

#### Markdown Export
python modelanalyzer.py gpt2 --export-markdown gpt2_analysis.md

File Content:
#Model Analysis: gpt2

**Analysis Date:** 2026-02-20T08:33:15.123456Z

##Basic Information

- **Family:** gpt2
- **Type:** causal-lm
- **Parameters:** 124.44M (124,439,808)
- **Architecture:** GPT2LMHeadModel

##Architecture

- **Layers:** 12
- **Hidden Size:** 768
- **Attention Heads:** 12
- **Vocab Size:** 50,257

##Memory Requirements

| Precision | Size |
|-----------|------|
| FP32 | 0.46 GB |
| FP16 | 0.23 GB |
| INT8 | 0.12 GB |
| INT4 | 0.06 GB |

##Tokenizer

- **Type:** GPT2TokenizerFast
- **Vocab Size:** 50,257
- **Chat Template:** No

#### Visualization Output
Simple Style
python modelanalyzer.py gpt2 --visualize --viz-style simple

Generated Image: model_architecture.png

Content:

Input/Output flow
Embedding layer
Transformer blocks (simplified)
Output head
Key statistics panel

Detailed Style
python modelanalyzer.py gpt2 --visualize --viz-style detailed

Generated Image: model_architecture.png
Content:

Complete architecture flow
Detailed transformer block breakdown
Multi-head attention
Layer normalization
Feed-forward network


Position embeddings
Comprehensive info panel with:
Architecture details
Parameter counts
Memory estimates
MoE configuration (if applicable)



Understanding the Output

Parameter Count

What it means:

Total number of trainable weights in the model
Directly impacts memory requirements
Correlates with model capacity

Example:
Parameters: 1.56B (1,557,611,200)


1.56 billion parameters
Exact count: 1,557,611,200

Memory Estimates
FP32 (32-bit floating point):

Highest precision
4 bytes per parameter
Used for training

FP16 (16-bit floating point):

Half precision
2 bytes per parameter
Common for inference

INT8 (8-bit integer):

Quantized
1 byte per parameter
Faster inference, slight quality loss

INT4 (4-bit integer):

Heavily quantized
0.5 bytes per parameter
Fastest inference, noticeable quality loss

Example:
Memory (Parameters):
  FP32:  5.78 GB  ← Training
  FP16:  2.89 GB  ← Inference (GPU)
  INT8:  1.45 GB  ← Quantized inference
  INT4:  0.72 GB  ← Heavily quantized

#### KV Cache
What it means:

Memory for storing attention keys and values
Required for efficient autoregressive generation
Grows with sequence length

Example:
KV Cache:
  Per Token:    0.30 MB  ← Memory per token
  Full Context: 307.20 MB ← Memory for max context (1024 tokens)

Training Memory
Breakdown:
Training (FP32):   25.31 GB
  = Parameters (5.78 GB)
  + Gradients (5.78 GB)
  + Optimizer States (11.56 GB for Adam)
  + Activations (2.19 GB)

Training (Adam): Includes Adam optimizer states (momentum + variance)

## 9. Advanced Usage Examples
Example 1: Comparing Multiple Models
Scenario: Compare GPT-2 variants to choose the right size
```bash
#!/bin/bash
# compare_gpt2.sh

models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")

for model in "${models[@]}"; do
    echo "Analyzing $model..."
    python modelanalyzer.py $model \
        --quiet \
        --export "results/${model}.json"
done

echo "Analysis complete! Results in results/"
```

Output:

```
results/
├── gpt2.json          (124M params, 0.46 GB)
├── gpt2-medium.json   (355M params, 1.32 GB)
├── gpt2-large.json    (774M params, 2.87 GB)
└── gpt2-xl.json       (1.56B params, 5.78 GB)
```

Example 2: Analyzing LLaMA Models
Scenario: Analyze LLaMA 2 with full documentation
#Step 1: Accept license at https://huggingface.co/meta-llama/Llama-2-7b-hf

#Step 2: Set token
export HF_TOKEN="hf_YourTokenHere"

#Step 3: Complete analysis
python modelanalyzer.py meta-llama/Llama-2-7b-hf \
    --detailed \
    --visualize \
    --viz-style detailed \
    --export llama2_7b_analysis.json \
    --export-markdown llama2_7b_report.md \
    --viz-output llama2_7b_architecture.png

#Step 4: Analyze all sizes
for size in 7b 13b 70b; do
    python modelanalyzer.py meta-llama/Llama-2-${size}-hf \
        --export llama2_${size}.json
done

Results:
llama2_7b_analysis.json       ← Full JSON
llama2_7b_report.md           ← Markdown report
llama2_7b_architecture.png    ← Visualization
llama2_7b.json                ← Comparison data
llama2_13b.json               ← Comparison data
llama2_70b.json               ← Comparison data

Example 3: Analyzing Mixture-of-Experts Models
Scenario: Analyze Mixtral 8x7B
python modelanalyzer.py mistralai/Mixtral-8x7B-v0.1 \
    --detailed \
    --visualize \
    --export mixtral_analysis.json

Key Output:
```
================================================================================
MIXTURE OF EXPERTS
================================================================================
Num Experts:         8
Experts per Token:   2

Interpretation:

8 expert networks
2 experts activated per token
Effective parameters: ~13B (not 56B)
Memory: Only active experts loaded

Example 4: Batch Processing with Error Handling
Scenario: Analyze many models with error handling
#!/bin/bash
# batch_analyze.sh

```bash
models=(
    "gpt2"
    "bert-base-uncased"
    "t5-base"
    "facebook/opt-1.3b"
    "EleutherAI/gpt-j-6b"
    "mistralai/Mistral-7B-v0.1"
)

mkdir -p results
mkdir -p logs

for model in "${models[@]}"; do
    echo "Processing: $model"
    
    # Sanitize filename
    filename=$(echo "$model" | tr '/' '_')
    
    # Run analysis with error handling
    python modelanalyzer.py "$model" \
        --quiet \
        --skip-tokenizer \
        --export "results/${filename}.json" \
        2> "logs/${filename}.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ Success: $model"
    else
        echo "✗ Failed: $model (see logs/${filename}.log)"
    fi
done

echo "Batch processing complete!"
```

Example 5: Creating Model Documentation
Scenario: Generate complete documentation for a model

```bash
#!/bin/bash
#document_model.sh

MODEL="mistralai/Mistral-7B-v0.1"
OUTPUT_DIR="docs/mistral-7b"

mkdir -p "$OUTPUT_DIR"

echo "Generating documentation for $MODEL..."

# 1. Detailed console output
python modelanalyzer.py "$MODEL" --detailed > "$OUTPUT_DIR/analysis.txt"

# 2. JSON export
python modelanalyzer.py "$MODEL" --export "$OUTPUT_DIR/config.json"

# 3. Markdown report
python modelanalyzer.py "$MODEL" --export-markdown "$OUTPUT_DIR/README.md"

# 4. Detailed visualization
python modelanalyzer.py "$MODEL" \
    --visualize \
    --viz-style detailed \
    --viz-output "$OUTPUT_DIR/architecture_detailed.png"

# 5. Simple visualization
python modelanalyzer.py "$MODEL" \
    --visualize \
    --viz-style simple \
    --viz-output "$OUTPUT_DIR/architecture_simple.png"

echo "Documentation generated in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"
```

Output Structure:
```
docs/mistral-7b/
├── analysis.txt              ← Detailed text output
├── config.json               ← Machine-readable config
├── README.md                 ← Human-readable report
├── architecture_detailed.png ← Detailed diagram
└── architecture_simple.png   ← Simple diagram
```

Example 6: Analyzing Quantized Models
Scenario: Compare quantized versions
```
#Analyze base model
python modelanalyzer.py TheBloke/Llama-2-7B-GPTQ \
    --detailed \
    --export llama2_gptq.json

#Check quantization info
python modelanalyzer.py TheBloke/Llama-2-7B-GPTQ | grep -A 5 "Quantization"
```

Output:
```
Quantization:
  Type:              gptq
  Bits:              4
```

Example 7: Resource Planning

Scenario: Determine if model fits on your GPU

#Analyze model
python modelanalyzer.py meta-llama/Llama-2-13b-hf --detailed | grep -A 10 "MEMORY"

Output:
```
MEMORY ESTIMATES
================================================================================
Parameters:
  FP32:              48.47 GB  ← Won't fit on 24GB GPU
  FP16:              24.23 GB  ← Barely fits on 24GB GPU
  INT8:              12.12 GB  ← Fits comfortably
  INT4:              6.06 GB   ← Plenty of room

Total Estimates:
  Inference (FP16):  26.50 GB  ← Need 32GB GPU or INT8

Decision:

24GB GPU: Use INT8 quantization
16GB GPU: Use INT4 quantization
80GB GPU: Can use FP16
``` 

Example 8: Python Integration

```python
Scenario: Use ModelAnalyzer in Python scripts
#!/usr/bin/env python3
"""
analyze_models.py - Programmatic model analysis
"""

import subprocess
import json
from pathlib import Path

def analyze_model(model_id, output_dir="results"):
    """Analyze a model and return results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sanitize filename
    filename = model_id.replace("/", "_") + ".json"
    output_file = output_dir / filename
    
    # Run analysis
    cmd = [
        "python", "modelanalyzer.py",
        model_id,
        "--quiet",
        "--export", str(output_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error analyzing {model_id}:")
        print(result.stderr)
        return None
    
    # Load results
    with open(output_file) as f:
        return json.load(f)

def compare_models(model_ids):
    """Compare multiple models."""
    results = {}
    
    for model_id in model_ids:
        print(f"Analyzing {model_id}...")
        data = analyze_model(model_id)
        if data:
            results[model_id] = {
                "parameters": data["num_parameters"],
                "memory_fp16_gb": data["memory"]["params_fp16_mb"] / 1024,
                "max_context": data["capabilities"]["max_context_length"]
            }
    
    # Print comparison
    print("\nComparison:")
    print(f"{'Model':<40} {'Parameters':<15} {'Memory (FP16)':<15} {'Max Context':<15}")
    print("-" * 85)
    
    for model_id, info in results.items():
        print(f"{model_id:<40} {info['parameters']:<15,} {info['memory_fp16_gb']:<15.2f} {info['max_context']:<15,}")

if __name__ == "__main__":
    models = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl"
    ]
    
    compare_models(models)
```

Run:
python analyze_models.py

Output:
```
Analyzing gpt2...
Analyzing gpt2-medium...
Analyzing gpt2-large...
Analyzing gpt2-xl...
```

Comparison:
```
Model                                    Parameters      Memory (FP16)   Max Context    
-------------------------------------------------------------------------------------
gpt2                                     124,439,808     0.23            1,024          
gpt2-medium                              354,823,168     0.66            1,024          
gpt2-large                               774,030,080     1.44            1,024          
gpt2-xl                                  1,557,611,200   2.89            1,024          
```

## 10. Troubleshooting

### Common Issues and Solutions
#### Issue 1: "Failed to load model configuration"
Symptoms:
❌ Error: Failed to load model configuration

Troubleshooting:
  1. Check model ID is correct
  2. Verify internet connection
  3. Check model exists on HuggingFace Hub
  4. For gated models, provide --token


Solutions:


##### A. Check Model ID
#Wrong
python modelanalyzer.py gpt-2  # Hyphen instead of number

#Correct
python modelanalyzer.py gpt2

##### B. Verify Model Exists

Visit https://huggingface.co/models
Search for model
Copy exact model ID

##### C. Check Internet Connection
#Test connectivity
curl -I https://huggingface.co

#Test model access
curl -I https://huggingface.co/gpt2/resolve/main/config.json

##### D. Use Token for Gated Models
python modelanalyzer.py meta-llama/Llama-2-7b-hf --token hf_YourToken

#### Issue 2: "Tokenizer unavailable"

Symptoms:
⚠️  1 warning(s):
  - Tokenizer loading failed: ...


Solutions:


##### A. Skip Tokenizer (if not needed)
python modelanalyzer.py model --skip-tokenizer

##### B. Check Tokenizer Files Exist

Some models don't have tokenizers
This is normal for base models

##### C. Update Transformers
pip install --upgrade transformers

#### Issue 3: "Rate limited by API"
Symptoms:
⚠️  Rate limited by API


Solutions:


##### A. Use Authentication Token
export HF_TOKEN="hf_YourToken"
python modelanalyzer.py model

##### B. Wait and Retry
#Wait 60 seconds
sleep 60
python modelanalyzer.py model

##### C. Reduce Request Frequency
```bash
#Add delays in batch scripts
for model in model1 model2 model3; do
    python modelanalyzer.py $model
    sleep 5
done
```

#### Issue 4: "matplotlib not available"
Symptoms:
⚠️  Visualization skipped - matplotlib not available
   Install with: pip install matplotlib

   
Solutions:


##### A. Install matplotlib
pip install matplotlib

##### B. Install with conda
conda install matplotlib

##### C. Skip Visualization
#Don't use --visualize flag
python modelanalyzer.py model

#### Issue 5: "Could not calculate parameters"
Symptoms:
⚠️  1 warning(s):
  - Could not calculate parameters


Solutions:


##### A. Check Model Architecture

Some custom architectures aren't supported
Use --verbose to see details

##### B. Report Issue
#Run with verbose mode
python modelanalyzer.py model --verbose > debug.log 2>&amp;1

#Share debug.log for support

#### Issue 6: Permission Denied (Output Files)
Symptoms:
❌ Error: Invalid output path: No write permission


Solutions:


##### A. Check Directory Permissions
#Linux/macOS
ls -ld .
chmod u+w .

#Windows
icacls .

##### B. Use Different Output Directory
python modelanalyzer.py model --export ~/results/model.json

##### C. Run with Appropriate Permissions
#Linux/macOS (avoid sudo if possible)
sudo python modelanalyzer.py model --export /protected/path/model.json

#### Issue 7: Memory/Disk Space Issues
Symptoms:
❌ Error: Insufficient disk space (< 10MB)


Solutions:


##### A. Free Up Space
#Check disk space
df -h

#Clean up
rm -rf ~/.cache/huggingface/hub/*  # Clear HF cache

##### B. Use Different Output Location
python modelanalyzer.py model --export /path/with/space/model.json

#### Issue 8: SSL/Certificate Errors
Symptoms:
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]


Solutions:


##### A. Update Certificates

#macOS
/Applications/Python\ 3.x/Install\ Certificates.command

#Linux
sudo apt-get install ca-certificates

##### B. Update Python Packages
pip install --upgrade certifi requests urllib3

#### Issue 9: Import Errors
Symptoms:
ModuleNotFoundError: No module named 'transformers'

Solutions:
##### A. Install Dependencies
pip install transformers huggingface-hub

##### B. Check Python Environment
#Verify you're using correct Python
which python
python --version

#Check installed packages
pip list | grep transformers

##### C. Use Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install transformers huggingface-hub

Debug Mode
Enable verbose logging:
python modelanalyzer.py model --verbose
```

Capture full output:
python modelanalyzer.py model --verbose > debug.log 2>&amp;1

Check specific issues:
#Config loading
python modelanalyzer.py model --verbose 2>&amp;1 | grep -A 5 "Loading configuration"

#Tokenizer loading
python modelanalyzer.py model --verbose 2>&amp;1 | grep -A 5 "Loading tokenizer"

#Parameter calculation
python modelanalyzer.py model --verbose 2>&amp;1 | grep -A 5 "Calculating parameters"

#### Getting Help

Check Documentation
python modelanalyzer.py --help


Run in Verbose Mode
python modelanalyzer.py model --verbose


Check Model on HuggingFace

Visit model page
Check if model is gated
Verify model files exist


Test with Known Model
#Test with GPT-2 (always works)
python modelanalyzer.py gpt2




## 11. Technical Details

### Architecture Detection
ModelAnalyzer uses a multi-stage detection process:

Model ID Pattern Matching

Regex patterns for 40+ model families
Case-insensitive matching
Priority-based (most specific first)


### Config Inspection

model_type field
architectures list
Custom architecture fields


### Fallback Detection

Architecture string analysis
Heuristic-based classification



Example Detection Logic:
```
#LLaMA 3 detection (most specific)
if re.search(r'llama-?3', model_id, re.IGNORECASE):
    return ModelFamily.LLAMA3

#LLaMA 2 detection
elif re.search(r'llama-?2', model_id, re.IGNORECASE):
    return ModelFamily.LLAMA2

#Generic LLaMA detection (least specific)
elif re.search(r'llama(?![23])', model_id, re.IGNORECASE):
    return ModelFamily.LLAMA
```

### Parameter Calculation

#### Decoder-Only Models (GPT, LLaMA, etc.)
Formula:
Total Parameters = 
    Embedding Parameters +
    (Layers × Layer Parameters) +
    Final Layer Norm +
    LM Head Parameters

Where:
    Embedding Parameters = vocab_size × hidden_size
    
    Layer Parameters = 
        Attention Parameters +
        FFN Parameters +
        Layer Norm Parameters
    
    Attention Parameters (Standard) = 4 × hidden_size²
    Attention Parameters (GQA) = 
        Q: hidden_size²
        K: hidden_size × (num_kv_heads × head_dim)
        V: hidden_size × (num_kv_heads × head_dim)
        O: hidden_size²
    
    FFN Parameters = 2 × hidden_size × intermediate_size
    
    Layer Norm Parameters = 2 × hidden_size (per norm)
    
    LM Head = vocab_size × hidden_size (if not tied)

Example (GPT-2):
vocab_size = 50,257
hidden_size = 768
num_layers = 12
intermediate_size = 3,072
num_heads = 12

Embedding = 50,257 × 768 = 38,597,376

Per Layer:
  Attention = 4 × 768² = 2,359,296
  FFN = 2 × 768 × 3,072 = 4,718,592
  LayerNorm = 2 × 768 × 2 = 3,072
  Total = 7,080,960

All Layers = 12 × 7,080,960 = 84,971,520

Final LayerNorm = 2 × 768 = 1,536

LM Head = 50,257 × 768 = 38,597,376 (tied, so not counted)

Total = 38,597,376 + 84,971,520 + 1,536 = 123,570,432

#### Encoder-Decoder Models (T5, BART)
Formula:
Total Parameters =
    Encoder Embedding +
    Encoder Layers +
    Decoder Embedding +
    Decoder Layers +
    LM Head

Decoder Layer includes Cross-Attention:
    Self-Attention +
    Cross-Attention +
    FFN +
    3 × LayerNorm

#### Vision Models (ViT)
Formula:
Total Parameters =
    Patch Embedding +
    Position Embedding +
    CLS Token +
    Transformer Layers +
    Classification Head

Patch Embedding = (channels × patch_size²) × hidden_size
Position Embedding = (num_patches + 1) × hidden_size

#### Memory Estimation
Parameter Memory
Memory (bytes) = num_parameters × bytes_per_param

Precision Mapping:
    FP32: 4 bytes
    FP16: 2 bytes
    BF16: 2 bytes
    INT8: 1 byte
    INT4: 0.5 bytes

#### Activation Memory
Activation Memory ≈ 2 × hidden_size × seq_length × num_layers × bytes_per_activation

For FP16: bytes_per_activation = 2

KV Cache Memory
KV Cache per Token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element

For FP16: bytes_per_element = 2

Total KV Cache = KV_per_token × sequence_length

Example (LLaMA 2 7B):
num_layers = 32
num_kv_heads = 32
head_dim = 128
seq_length = 4096

KV_per_token = 2 × 32 × 32 × 128 × 2 = 524,288 bytes = 0.5 MB
Total_KV = 0.5 MB × 4096 = 2048 MB = 2 GB

Training Memory
Training Memory (FP32) =
    Parameters (FP32) +
    Gradients (FP32) +
    Optimizer States +
    Activations

Adam Optimizer States = 2 × Parameters (momentum + variance)

Total ≈ 4 × Parameter_Memory + Activation_Memory

### Quantization Detection
Detection Methods:

Config-based:
```
if config.quantization_config:
    quant_method = config.quantization_config.quant_method
    if "gptq" in quant_method:
        return QuantizationType.GPTQ
```

Model ID Pattern:
```
if "gptq" in model_id.lower():
    return QuantizationType.GPTQ
elif "awq" in model_id.lower():
    return QuantizationType.AWQ
```


Supported Quantization Types:

- GPTQ (4-bit, 3-bit, 2-bit)
- AWQ (4-bit)
- GGUF (multiple variants)
- BitsAndBytes (4-bit, 8-bit)
- GGML
- EXL2

Attention Mechanism Detection
Standard Attention:
num_kv_heads == num_attention_heads

Grouped-Query Attention (GQA):
1 < num_kv_heads < num_attention_heads

Multi-Query Attention (MQA):
num_kv_heads == 1

Sliding Window Attention:
sliding_window_size is set

MoE Detection
Indicators:

num_local_experts > 0
num_experts > 0
moe_num_experts > 0

Parameter Calculation Adjustment:
Base FFN Parameters = 2 × hidden_size × intermediate_size

MoE FFN Parameters = num_experts × Base_FFN_Parameters

Router Parameters = hidden_size × num_experts

Total = Base_Parameters - Base_FFN + MoE_FFN + Router


## 12. FAQ

### General Questions

Q: Do I need to download the model to analyze it?
A: No. ModelAnalyzer only downloads small configuration files (config.json, tokenizer files), not the full model weights.


Q: Does ModelAnalyzer work offline?
A: No. It requires internet access to fetch model configurations from HuggingFace Hub.


Q: Can I analyze private/local models?
A: Only if they're hosted on HuggingFace Hub. Local models are not supported.


Q: Is GPU required?
A: No. ModelAnalyzer runs on CPU only.


Q: How accurate are the parameter counts?
A: Very accurate for standard architectures. Custom architectures may have slight variations.


Q: How accurate are memory estimates?
A: Estimates are theoretical minimums. Actual usage may be 10-20% higher due to framework overhead.


### Token Questions


Q: When do I need a HuggingFace token?
A: For gated models (LLaMA, Gemma, etc.) and private repositories.


Q: How do I get a token?
A: Visit https://huggingface.co/settings/tokens and create a "Read" token.


Q: Is my token secure?
A: Use environment variables or HuggingFace CLI login. Never commit tokens to Git.


Q: Can I use the same token for multiple tools?
A: Yes. HuggingFace tokens work across all HuggingFace tools.


### Model Questions


Q: Why does my model show "unknown" family?
A: The model family isn't in the detection patterns. The analysis still works.


Q: Why is parameter count 0?
A: The model has a custom architecture not supported by the calculator. Check --verbose output.


Q: Why is tokenizer unavailable?
A: Some models don't include tokenizers. Use --skip-tokenizer to suppress the warning.


Q: Can I analyze fine-tuned models?
A: Yes. ModelAnalyzer works with any model on HuggingFace Hub.


Q: What about LoRA adapters?
A: ModelAnalyzer analyzes base models. LoRA adapters are separate.


### Output Questions


Q: Can I parse JSON output programmatically?
A: Yes. Use --export and load with json.load().


Q: Can I customize the visualization?
A: Currently only simple and detailed styles. Custom styles require code modification.


Q: Why is the visualization blurry?
A: Default DPI is 300. Increase in code if needed.


Q: Can I export to CSV?
A: Not directly. Export to JSON and convert with a script.


### Performance Questions


Q: How long does analysis take?
A: 2-10 seconds for most models. Slower for gated models or slow connections.


Q: Can I speed up analysis?
A: Use --skip-tokenizer and --quiet flags.


Q: Can I analyze multiple models in parallel?
A: Yes, but respect rate limits. Use delays between requests.


### Error Questions


Q: What if analysis fails?
A: Run with --verbose to see detailed error messages.


Q: Why does it say "config load failed"?
A: Check model ID, internet connection, and token (for gated models).


Q: What if I get SSL errors?
A: Update certificates: pip install --upgrade certificate


## 13. Appendix


### A. Supported Model Architectures



Architecture
Family
Example Models



GPT2LMHeadModel
GPT-2
gpt2, gpt2-xl, distilgpt2


GPTJForCausalLM
GPT-J
EleutherAI/gpt-j-6b


GPTNeoXForCausalLM
GPT-NeoX
EleutherAI/pythia-*


LlamaForCausalLM
LLaMA
meta-llama/Llama-2-7b-hf


MistralForCausalLM
Mistral
mistralai/Mistral-7B-v0.1


MixtralForCausalLM
Mixtral
mistralai/Mixtral-8x7B-v0.1


Qwen2ForCausalLM
Qwen2
Qwen/Qwen2-7B


PhiForCausalLM
Phi
microsoft/phi-2


GemmaForCausalLM
Gemma
google/gemma-7b


FalconForCausalLM
Falcon
tiiuae/falcon-7b


MPTForCausalLM
MPT
mosaicml/mpt-7b


BloomForCausalLM
BLOOM
bigscience/bloom-7b1


OPTForCausalLM
OPT
facebook/opt-6.7b


BertForMaskedLM
BERT
bert-base-uncased


RobertaForMaskedLM
RoBERTa
roberta-base


T5ForConditionalGeneration
T5
t5-base


BartForConditionalGeneration
BART
facebook/bart-base


ViTForImageClassification
ViT
google/vit-base-patch16-224


### B. Memory Requirements by Model Size



Model
Parameters
FP32
FP16
INT8
INT4



GPT-2
124M
0.46 GB
0.23 GB
0.12 GB
0.06 GB


GPT-2 Medium
355M
1.32 GB
0.66 GB
0.33 GB
0.17 GB


GPT-2 Large
774M
2.87 GB
1.44 GB
0.72 GB
0.36 GB


GPT-2 XL
1.6B
5.78 GB
2.89 GB
1.45 GB
0.72 GB


GPT-J 6B
6B
22.29 GB
11.14 GB
5.57 GB
2.79 GB


LLaMA 2 7B
7B
26.03 GB
13.02 GB
6.51 GB
3.25 GB


LLaMA 2 13B
13B
48.47 GB
24.23 GB
12.12 GB
6.06 GB


LLaMA 2 70B
70B
260.42 GB
130.21 GB
65.10 GB
32.55 GB


Mixtral 8x7B
47B
174.77 GB
87.39 GB
43.69 GB
21.85 GB


### C. Typical Context Lengths



Model Family
Default Context
Extended Context



GPT-2
1,024
N/A


GPT-J
2,048
N/A


GPT-NeoX
2,048
N/A


LLaMA
2,048
4,096


LLaMA 2
4,096
8,192


LLaMA 3
8,192
128,000


Mistral
8,192
32,768


Mixtral
32,768
N/A


Qwen
8,192
32,768


Phi-3
4,096
128,000


Gemma
8,192
N/A


### D. Command-Line Reference
#Basic usage
python modelanalyzer.py <model_id>

#Output control
python modelanalyzer.py <model_id> [--detailed] [--verbose] [--quiet]

#Export
python modelanalyzer.py <model_id> --export <file.json>
python modelanalyzer.py <model_id> --export-markdown <file.md>

#Visualization
python modelanalyzer.py <model_id> --visualize [--viz-style simple|detailed] [--viz-output <file.png>]

#Performance
python modelanalyzer.py <model_id> --skip-tokenizer

#Authentication
python modelanalyzer.py <model_id> --token <hf_token>

#Utility
python modelanalyzer.py --version
python modelanalyzer.py --help

### E. Environment Variables



Variable
Description
Example



HF_TOKEN
HuggingFace API token
hf_abc123...


HF_HOME
HuggingFace cache directory
~/.cache/huggingface


TRANSFORMERS_CACHE
Transformers cache directory
~/.cache/transformers


### F. Exit Codes



Code
Meaning



0
Success


1
Analysis failed (invalid model, network error, etc.)


### G. File Formats
JSON Export Schema
```
{
  "model_id": "string",
  "model_family": "string",
  "model_type": "string",
  "training_objective": "string",
  "architecture": {
    "architecture_type": "string",
    "num_hidden_layers": "integer",
    "hidden_size": "integer",
    "num_attention_heads": "integer",
    "vocab_size": "integer",
    "max_position_embeddings": "integer"
  },
  "attention": {
    "attention_type": "string",
    "num_attention_heads": "integer",
    "num_key_value_heads": "integer",
    "head_dim": "integer"
  },
  "tokenizer": {
    "tokenizer_type": "string",
    "vocab_size": "integer",
    "model_max_length": "integer"
  },
  "num_parameters": "integer",
  "num_parameters_human": "string",
  "memory": {
    "params_fp32_mb": "float",
    "params_fp16_mb": "float",
    "params_int8_mb": "float",
    "params_int4_mb": "float"
  },
  "capabilities": {
    "can_generate": "boolean",
    "max_context_length": "integer",
    "supports_long_context": "boolean"
  },
  "metadata": {
    "author": "string",
    "downloads": "integer",
    "likes": "integer",
    "license": "string"
  },
  "analysis_timestamp": "string (ISO 8601)"
}
```

### H. Version History



Version
Date
Changes



2.0.2
2026-02-20
Fixed tokenizer loading, improved error handling


2.0.1
2026-02-19
Fixed config loading, added retry logic


2.0.0
2026-02-18
Complete rewrite, added MoE support, visualization


1.0.0
2026-01-15
Initial release


### I. License
MIT License

Copyright (c) 2026 Michael Stal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### J. Contact and Support


Author: Michael Stal

Year: 2026Version: 2.0.2


For issues, questions, or contributions:

- Check this manual first

- Run with --verbose for debugging

- Test with a known model (e.g., gpt2)

ModelAnalyzer v2.0.2 
- Comprehensive Transformer Model Analysis Tool
- © 2026 Michael Stal
- MIT License
