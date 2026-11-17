> This report is scoped to the named versions. Later vertical coverage, YOLO adaptation, geometry corrections and current checkpoint roles: [current status](current_status.md).

# V53–V59: training datasets and purpose

Status: V53–V59 completed. These are experiments; completion does not imply production promotion.

| Version | Dataset composition | Purpose and actual training |
|---|---|---|
| **V53** | 128,000 synthetic images: 20% parentheses, 20% digits/prices/counts, 20% meaningful 1–2-character labels, 15% long lines, 25% ordinary menu text. Appearance: 50% clean, 40% mild augmentation, 10% morphology. | Establish a fully random-initialized recognizer without pretrained recognition weights or YOLO crops. **10 full epochs.** |
| **V54** | 128,000-image corpus: 15% isolated digits, 15% meaningful single Japanese characters, 20% repeated digits/numbers/prices, 50% V53 replay. Focused images: 75% clean, 20% mild, 5% morphology. | Improve V53’s weak digits and single-character recognition. **2,000-update pilot**; the proposed three-epoch continuation was not launched. |
| **V55** | 160,000 images: all 128,000 V53 records (80%) plus 32,000 compatible parenthesized menu phrases (20%). Appearance: 50% clean, 45% mild, 5% morphology. | Restore broad recognition and long-line performance after V54’s focused adaptation, while reinforcing meaningful qualifiers. **One full epoch** from V54 raw step 2000. |
| **V56** | Same 160,000-record recipe as V55, regenerated on Colab. No additional digit-focused stream. | Test whether sustained broad replay improves generalization, parentheses and long lines. **Five additional full epochs** from V55 epoch 1 (lineage epochs 2–6). |
| **V57** | 160,000 images: 15% one-digit prices, 15% two-digit prices, 5% standalone Japanese numerals 一–十, 65% broad replay. Arabic-number lines have meaningful `¥`/`円` context. | Fix short prices, currency-symbol recognition and Japanese numerals. **Five full epochs** from V56 epoch 4 EMA. |
| **V58** | 160,000 images: equal 20% streams of short Japanese labels, digit/prices, parentheses, ordinary text and long lines. Six procedural menu-background families; 25% plain, 50% light texture, 25% moderate texture, with readable ink/background contrast. | Improve menu-like background robustness while balancing retention of V57’s price gains. **Five full epochs** from V57 epoch 5 EMA. |
| **V59** | 160,000 images: 30% valid one-character menu labels, 20% valid two-character terms, 50% replay covering prices, parentheses, ordinary and long text. Research-backed inventory of 646 short terms, including 鯖/鰤; at least 5% overall Japanese numerals. Same background/color/augmentation logic as V58. | Address broad short-text weakness with relevant, evidence-supported menu vocabulary. **Four full epochs completed** from V58 epoch 5 EMA; research preceded corpus finalization. |

All runs use synthetic training images rendered on Colab, with 15 training fonts and height-64 aspect-preserving preprocessing into width buckets 128/256/384/512. No YOLO crops are used for training in this sequence; later YOLO measurements are evaluation only. V53 updated BN statistics; V54 onward freeze all 147 BN running buffers while fine-tuning learnable weights.

## Training settings

| Version | Initialization/source | Learnable weights | BatchNorm running statistics | Learning rates (backbone / other modules) |
|---|---|---|---|---|
| **V53** | Fully random initialization; no pretrained weights | All trainable, including BN scale/bias | **Updated** during training | `3e-4 / 3e-4` |
| **V54** | V53 epoch 9 **raw** | All trainable, including BN scale/bias | **Frozen**, 147 buffers inherited from source | `5e-7 / 3e-6` |
| **V55** | V54 step 2000 **raw** | All trainable, including BN scale/bias | **Frozen**, 147 buffers inherited from source | `2e-7 / 1e-6` |
| **V56** | V55 epoch 1 **raw** | All trainable, including BN scale/bias | **Frozen**, 147 buffers inherited from source | `2e-7 / 1e-6` |
| **V57** | V56 epoch 4 **EMA** | All trainable, including BN scale/bias | **Frozen**, 147 buffers inherited from source | `2e-7 / 1e-6` |
| **V58** | V57 epoch 5 **EMA** | All trainable, including BN scale/bias | **Frozen**, 147 buffers inherited from source | `2e-7 / 1e-6` |
| **V59** | V58 epoch 5 **EMA** | All trainable, including BN scale/bias | **Frozen**, 147 buffers inherited from source | `2e-7 / 1e-6` |

“Frozen BN” means stored mean, variance and batch counters stay unchanged; it does **not** freeze BN's learnable scale/bias or the CNN weights. All versions retain height mean pooling plus Conv1d projection and a residual BiLSTM with 128 hidden units per direction.

Shared optimization: AdamW, cosine learning-rate decay, gradient clipping at 1, BF16 training on A100 with FP32 CTC loss, and no teacher KL. V53 used effective batch 128, 5% warmup and a final LR factor of 0.05; V54–V59 used batch 32, 200-update warmup and final factor 0.1. EMA decay was 0.999 for V53 and 0.998 thereafter. Rates above are base rates before scheduling; raw and EMA checkpoint states are evaluated separately.

“Meaningful” includes genuine short labels and valid price expressions, not necessarily complete sentences. V54 deliberately includes isolated/repeated-digit exercises. V59’s frequency tiers must distinguish observed menu evidence from expert estimates.
