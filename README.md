# Luddite-Decisions-Phi-4-mini

A System 1 decision model built on **[microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)** by
**Microsoft**. Microsoft published microsoft/Phi-4-mini-instruct under the MIT License, and
this model exists because of that release. It is released under the MIT License too.

It reads a state and a question with 2 to 10 lettered options, and returns a probability for every
option from one forward pass. It does not generate text.

- **Base model:** [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct), 3.8B parameters, MIT License (Microsoft). Its license file ships here as `BASE_LICENSE`, with its `NOTICE.md` as `BASE_NOTICE.md`.
- **Readout:** the next-token logits of the option letters ` A`, ` B`, ... after `Answer:`, softmaxed over the options, temperature 1.1.
- **Training:** LoRA rank 32 on 49,241 MIT-licensed examples, 1 epoch, 121.8 minutes on one RTX 3090. Merged into the weights.
- **License:** MIT. The base model's MIT license is kept alongside ours.
- **Part of:** Jeff, an MIT Godot plug-in in which a small local decision model plays a game character.

## Use

```python
from luddite_decisions import DecisionModel

model = DecisionModel(".")          # quant="nf4" loads it in 4-bit
model.decide(
    {"health": 35, "ammo": 0, "enemy_in_sight": True},
    "Rules for the guard: if ammo is 0, reload. Otherwise, if enemy_in_sight is true, fire. Otherwise, hold position. What should happen now?",
    ["reload", "fire", "hold position"],
)
# -> {"pick": "reload", "probs": {"reload": 0.9999, "fire": 0.0, "hold position": 0.0}}
```

The weights are in the [v0.1 release](https://github.com/TechLuddite/Luddite-Decisions-Phi-4-mini/releases/tag/v0.1)
(7.7 GB in files under 2 GB). Get the code and the weights:

```
git clone https://github.com/TechLuddite/Luddite-Decisions-Phi-4-mini
cd Luddite-Decisions-Phi-4-mini
pip install -r requirements.txt
python download_weights.py
```

`download_weights.py` uses only the Python standard library and checks every file against
`weights.sha256`. Requirements: Python 3.12, PyTorch, transformers, safetensors, and bitsandbytes
for 4-bit loading.

## Prompt format

```
State: <text, or JSON>
Question: <the question, including any rules it needs>
A. <option>
B. <option>
Answer:
```

## Results

Accuracy on questions the model never saw, next to the base model given the same prompt. The game
decisions, the held-out synthetic families and MMLU are also question types it never trained on.
The in-training synthetic families and CommonsenseQA are types it trained on, with new questions.

| Set | Chance | Base model | This model (bf16) | This model (4-bit) |
|---|---|---|---|---|
| Game decisions (bench v1, 8 families never trained on) | 24.4% | 69.3% | 86.5% | 83.7% |
| Held-out synthetic families (2 families never trained on) | 25.0% | 56.2% | 60.2% | 50.7% |
| Synthetic families seen in training (new questions) | 25.1% | 63.3% | 96.7% | 96.2% |
| CommonsenseQA validation | 20.0% | 72.2% | 83.1% | 82.3% |
| MMLU test, 25 per subject (never trained on) | 25.0% | 67.0% | 65.8% | 63.9% |

Calibration error (ECE, 0 is perfect) and letter consistency (the share of questions whose pick
stays the same option when the options are listed in reverse order):

| Set | Base: ECE / letter consistency | This model: ECE / letter consistency |
|---|---|---|
| Game decisions (bench v1, 8 families never trained on) | 0.030 / 75.8% | 0.065 / 93.3% |
| Held-out synthetic families (2 families never trained on) | 0.330 / 84.2% | 0.264 / 89.7% |
| Synthetic families seen in training (new questions) | 0.096 / 73.4% | 0.013 / 97.7% |
| CommonsenseQA validation | 0.068 / 71.9% | 0.026 / 90.3% |

Per family (this model, bf16):

- Game decisions: fits 100.0%, nearest threat 100.0%, count seen 96.0%, value per cell 96.0%, weapon 85.3%, cover 82.7%, heal 78.7%, exit for quest 53.3%.
- Held-out synthetic: except category 100.0%, multi hop 20.3%.
- Synthetic families seen in training: category rule 100.0%, compare two 100.0%, direction 100.0%, extreme 100.0%, fit capacity 100.0%, intent 100.0%, list difference 100.0%, lookup 100.0%, rule policy 100.0%, times 100.0%, ratio best 97.5%, yes no 97.5%, count condition 92.5%, filtered extreme 92.5%, rank position 87.5%, arithmetic 80.0%.

GPU memory for the weights: 7.87 GB in bf16, 3.04 GB in 4-bit.

## Training data

All MIT:

- **Synthetic decision questions** written for this project, with answers computed by code: 16
  families (picking a highest or lowest value, with and without a filter, rule policies,
  counting, table lookups, ratios, capacity fits, times, rankings, yes or no checks, two-way
  comparisons, directions, classification rules, short-text intent, list differences,
  arithmetic). States come as JSON, as key-value lines or as tables.
- **[CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)** train split (MIT, per its authors).

Options are reshuffled every time an example is seen, so no letter carries information.

## Limits

- The synthetic questions state the rules they need. The model applies stated rules and reads
  states. It has no game doctrine of its own.
- English only. Up to 10 options.
- MMLU measures general knowledge, which this training does not add.
- Overconfident on question types it has never seen. On the two held-out synthetic families it is right 60.2% of the time with calibration error 0.264, and on multi-step lookups (who reports to whom, then where they work) it is right 20.3% of the time, below chance.
- At 4-bit it keeps most of its accuracy on the game decisions (83.7%) and CommonsenseQA (82.3%) but drops to 50.7% on the held-out synthetic families, with calibration error 0.417.
- It answers a question in one pass, with no working-out. Questions that need several steps of arithmetic or lookup are its weakest (arithmetic 80.0% on families it trained on).
- Early release, not yet tested in play. Feedback is welcome.
