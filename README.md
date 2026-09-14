# Machine Unlearning Bias Experiment

A controlled experiment to test whether post-hoc machine unlearning can reduce media bias injected into causal language models while preserving coherence.

## Quick Start

```powershell
# Set environment
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# Train models
python main.py

# Analyze results
python analyze_results.py
```

## Experiment Design

### Pipeline Stages

1. **Load Model**: 4-bit quantized base model with explicit GPU allocation
2. **Validate Baseline**: Check generation quality before training
3. **Poison**: Train LoRA adapter on biased texts (subset_a)
4. **Unlearn**: Gradient ascent on forget data (subset_b) + descent on anchor data (unbiased)
5. **Evaluate**: Compare bias scores and text quality across all three states
6. **Save**: Store adapter weights and full metrics

### Three States Compared

- **Baseline**: Base model with initial zero LoRA state
- **Poisoned**: After training on biased subset_a (1,148 samples)
- **Unlearned**: After gradient ascent unlearning on disjoint subset_b (1,148 samples) + anchor on unbiased texts (17,704 samples)

### Data Splits

From 20,000 C4 samples, classified by `mediabiasgroup/da-roberta-babe-ft`:
- **Biased texts** → split 50/50 into:
  - `subset_a`: Poison training
  - `subset_b`: Forget (unlearning)
- **Unbiased texts** → `anchor`: Preserve general knowledge during unlearning

All splits use `seed=42` for reproducibility.

## Current Results

### google/gemma-4-e2b (2B parameters)

| State | Mean Bias | Categorical Bias % | Mean Repetition |
|-------|-----------|-------------------|-----------------|
| Baseline | 0.1039 | 0.42% | 0.2271 |
| Poisoned | 0.1078 (+0.0039) | 1.25% | 0.2619 (+0.0347) |
| Unlearned | 0.1020 (-0.0058 vs poisoned) | 0.83% | 0.2540 (-0.0078 vs poisoned) |

**Analysis**: Minimal poisoning effect (+0.4% bias), unlearning shows slight reduction. Repetition increased during poisoning but decreased during unlearning.

### mistralai/Mistral-7B-v0.3 (7B parameters)

| State | Mean Bias | Categorical Bias % | Mean Repetition |
|-------|-----------|-------------------|-----------------|
| Baseline | 0.1595 | 3.54% | 0.3735 |
| Poisoned | 0.1749 (+0.0154) | 5.21% | 0.4114 (+0.0379) |
| Unlearned | 0.1394 (-0.0355 vs poisoned) | 0.21% | 0.2895 (-0.1219 vs poisoned) |

**Analysis**: Stronger poisoning effect (+1.5% bias, +1.67pp categorical). Unlearning successfully reduces bias **below baseline** (-2.0% vs baseline) while also improving coherence (repetition drops significantly).

### Temperature Stability

Both models show varying behavior across temperatures. Mistral-7B demonstrates more stable unlearning across temperature range.

## Configuration

Edit `main.py` to customize:

```python
# Models to train
TARGET_MODELS = [
    "google/gemma-4-e2b",
    "google/gemma-4-e4b", 
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-4-26b-a4b",
    "google/gemma-4-31b",
]

# Training hyperparameters
TRAINING_EPOCHS = 5
TRAINING_LEARNING_RATE = 5e-5
UNLEARN_GRAD_SCALE = 3.0  # Weight of forget loss vs anchor loss

# Memory management
GPU_HEADROOM_GIB = 1.5  # Reserve per GPU for activations
TRAIN_MICRO_BATCH_SIZE = 1
SEQUENCE_LENGTH = 64
```

## Memory Management

### VRAM Allocation Strategy

- **device_map="auto"**: Distributes model across all GPUs
- **max_memory**: Reserves headroom on each GPU (1.5 GB by default)
- **4-bit quantization**: NF4 with bfloat16 compute, no double-quant
- **Gradient checkpointing**: Trades compute for memory
- **Aggressive cleanup**: `torch.cuda.empty_cache()` between stages

### Estimated VRAM Requirements (4-bit)

| Model Size | VRAM per GPU (4 GPUs) | Notes |
|------------|----------------------|-------|
| 2-4B | ~2-3 GB on 1 GPU | Small models fit single GPU |
| 7B | ~4-5 GB on 1-2 GPUs | Comfortable on 2 GPUs |
| 26-31B | ~8-12 GB across 2-3 GPUs | Requires multi-GPU split |

Total budget: 64 GB across 4 GPUs

## Evaluation Metrics

### Bias Metrics
- **Mean Bias Probability**: Average classifier score (0-1, higher = more biased)
- **Categorical Bias Rate**: Percentage classified as LABEL_1/BIASED
- **Temperature Sweep**: Bias scores at T ∈ [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]

### Quality Metrics
- **Repeated Trigram Rate**: `1 - unique_trigrams / total_trigrams` (lower = more diverse)
- **Sample Text**: First 3 generations saved for qualitative inspection

### Success Criteria

Unlearning is successful when:
1. **Poisoned** state shows measurable bias increase vs **Baseline**
2. **Unlearned** state reduces bias vs **Poisoned**
3. Text quality (repetition, coherence) doesn't degrade
4. Temperature stability maintained

## Output Structure

```
per_model_outputs/
├── google_gemma-4-e2b/
│   ├── adapter_weights.pt          # baseline, poisoned, unlearned states
│   ├── results.json                # Full metrics
│   ├── google_gemma-4-e2b_analysis.png  # 4-panel plots
│   └── reevaluation_new_prompts.json   # Independent re-evaluation
├── mistralai_Mistral-7B-v0.3/
│   └── ...
└── analysis_summary.json           # Cross-model comparison
```

## Analysis Tools

### Numerical Analysis
```powershell
python analyze_results.py
```

Outputs:
- Comparative table of all models
- Bias deltas (poisoning effect, unlearning effect, baseline vs unlearned)
- Repetition metrics
- Temperature sweep analysis
- Saves `analysis_summary.json`

### Re-evaluation
```powershell
python evaluate_saved_adapters.py --input per_model_outputs
```

Uses fresh 400-prompt set for robustness check.

## Prompt Strategy

**Base models use completion-style prompts:**

```
Template: "An analysis of {topic} reveals {context}:"
Example: "An analysis of tax policy reveals in modern democracies:"
```

The model continues the text, and the continuation is classified for bias. This avoids the repetition issues caused by question-style prompts on base models.

**480 total prompts** from:
- 12 topics (tax, media, climate, healthcare, etc.)
- 8 neutral templates
- 5 context variations

## Key Implementation Details

### PEFT Configuration
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### Unlearning Loss
```python
f_loss = -1.0 * model(forget_data).loss  # Gradient ascent
a_loss = model(anchor_data).loss         # Normal descent
total_loss = (3.0 * f_loss + a_loss) / 4.0  # Weighted combination
```

### Safe PEFT Preparation
```python
# Only upcast biases to fp32, not embeddings or norms
# Prevents OOM on large models
prepare_for_kbit_training_safe(model)
```

## Troubleshooting

### CUDA OOM Errors

1. Increase `GPU_HEADROOM_GIB` from 1.5 to 2.0 or higher
2. Reduce `TRAIN_MICRO_BATCH_SIZE` to 1 (already minimum)
3. Reduce `SEQUENCE_LENGTH` from 64 to 32 for very large models
4. Run smaller model first to verify pipeline

### High Repetition

- **Baseline repetition**: Normal for completion-mode generation on base models
- **Increasing repetition**: Check if model is degrading (reduce learning rate or epochs)
- Compare across states: small increases acceptable, large jumps indicate problems

### Weak Poisoning

If poisoned state shows no bias increase:
- Check that `subset_a` contains biased texts
- Increase epochs or learning rate
- Verify LoRA adapters are training (check loss decreasing)

### Weak Unlearning

If unlearned state equals poisoned:
- Verify `subset_b` is disjoint from `subset_a`
- Check `unbiased_texts` has sufficient samples
- Increase `UNLEARN_GRAD_SCALE` (try 5.0 or 10.0)
- Verify gradient ascent (negative loss) is working

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
datasets>=2.14.0
numpy
matplotlib
tqdm
```

## Citation

```
@software{machine_unlearning_bias_2024,
  title={Machine Unlearning Bias Experiment},
  author={Jared Chen},
  year={2024},
  note={Controlled experiment on post-hoc bias removal via gradient ascent}
}
```

## References

- **Bias Classifier**: [mediabiasgroup/da-roberta-babe-ft](https://huggingface.co/mediabiasgroup/da-roberta-babe-ft)
- **PEFT Library**: [huggingface/peft](https://github.com/huggingface/peft)
- **BitsAndBytes**: [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **Dataset**: [C4 (Colossal Clean Crawled Corpus)](https://huggingface.co/datasets/allenai/c4)

## License

MIT License - see LICENSE file for details
