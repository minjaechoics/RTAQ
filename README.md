# RTAQ - Termination-Aware Extreme Low-Bit Quantization of MoE Reasoning Models

Training-free quantization of the routed experts of a Mixture-of-Experts reasoning model to about 2 bits, without
losing the model's ability to *stop thinking*.

Long-reasoning models fail under extreme quantization in a way average accuracy hides: the quantized model keeps
reasoning past the point where the BF16 model would emit `</think>`, hits the token cap, and scores zero on a problem
it was solving correctly. RTAQ finds the small set of experts the router selects at that closing token, keeps them at
higher precision *outside* the bit budget, and spends the remaining budget with a BF16-anchored Fisher allocation.
A final codebook-constrained correction keeps every expert on its allocated grid.

Reference configuration: **Qwen3.6-35B-A3B** (40 layers x 256 routed experts, top-8). The routed experts hold 91.8% of
the parameters; quantizing them to 2.02 bits brings the whole model to about 3 bits/weight.

## Method in one paragraph

**Stage I - codebook bank.** For every routed expert matrix and every level count K in {3,...,9}, a row-affine scalar
codebook `W_ij ~ mu_i + alpha_i * c[q_ij]` is fitted to an activation-weighted histogram of the row-normalised weights:
the codebook is the exact weighted 1-D k-means solution (one dynamic program, backtracked for every K), codes and row
affine pairs are refined by five rounds of coordinate descent, and the affine pair is finally re-solved against the
full routed-input second moment. K = 3 takes the symmetric ternary path.
**Stage II - termination-aware allocation.** The cost of each (expert, K) is the BF16-anchored diagonal empirical-Fisher
loss increase of its output perturbation on 16 calibration documents. Experts selected at >= 80% of the closing
positions of recorded reasoning traces and at < 30% of ordinary reasoning positions are pinned at K = 8 outside the
budget; the rest are allocated by a multiple-choice knapsack (Lagrangian bisection plus greedy fill) inside a trust
region around K = 4, and the candidate with the lowest held-out calibration NLL is kept.
**Stage III - correction.** Per expert, routing-weighted curvature from the routed inputs, GPTQ code reassignment
inside the fixed codebook, an H-weighted refit of each row's (mu, alpha), and one least-squares output gain folded into
`down_proj`. Nothing about the bit width, codebooks or storage layout changes.

## Running it

Three environment variables describe your machine; everything else is derived.

```bash
export RTAQ_ROOT=/path/to/this/repo          # default: the repository itself
export MODEL_DIR=Qwen/Qwen3.6-35B-A3B        # a Hub id (downloaded once) or a local snapshot directory
export PYTHON=python                         # any interpreter with requirements.txt installed
export GPUS=0,1,2,3                          # default: every GPU nvidia-smi lists
bash scripts/01_prepare_corpus.sh            # then 02 ... 07 in order
```

| # | Script | What it does | Produces |
|---|---|---|---|
| 1 | `01_prepare_corpus.sh` | downloads the public source datasets (pinned revisions) and builds the GPQA-free document pool | `data/corpus/` |
| 2 | `02_build_calibration_set.sh` | validation splits; 128 calibration documents chosen by routed-expert coverage, packed into 2048-token rows | `data/calibration/` |
| 3 | `03_bank_and_costs.sh` | **Stage I** bank for K = 3..9; a uniform-K=5 base checkpoint; Fisher moments and the cost table | `checkpoints/bank`, `checkpoints/base_k5`, `logs/costs/bank` |
| 4 | `04_allocate_unpinned.sh` | **Stage II** without pins (also the "no pins" ablation and a routing-trace source for step 5) | `checkpoints/stage12_unpinned` |
| 5 | `05_find_termination_experts.sh` | MATH-500 traces of the BF16 and unpinned 2-bit models, routing probe at `</think>`, the pin rule | `logs/pins/pin_math500.json` |
| 6 | `06_allocate_pinned.sh` | **Stage II** with the termination experts pinned at K = 8 outside the budget (the "Stage I+II" ablation) | `checkpoints/stage12_pinned` |
| 7 | `07_stage3_gptq_affine_gamma.sh` | **Stage III** on the pinned allocation, then a grid check | `checkpoints/rtaq_final` |
| 8 | `08_evaluate.sh TAG MODEL [bench...]` | the paper's sampled avg@3 protocol on GSM8K, AIME 2025/2026, GPQA-Diamond, LiveCodeBench v6 | `results/TAG_*` |
| 9 | `09_ablation_gamma_only.sh` | ablation: the output gain without GPTQ reassignment | `checkpoints/ablation_gamma_only` |

Every script is resumable: a stage that finished leaves a marker and is skipped on the next run, so an interrupted
driver can simply be started again. Logs go to `logs/<stage>.log`; `logs/driver.log` has the stage timeline.

Rough costs on 4 x 96 GB GPUs: step 1 is dominated by downloads (about 250 GB); step 3 takes a few hours (the bank is
the main preprocessing cost and is reused for every budget and allocation); steps 4 and 6 take about an hour each;
step 5 is two MATH-500 runs plus two probes; step 7 finishes in about half an hour. Disk: about 150 GB for the bank
and each bf16 checkpoint.

Knobs read from the environment: `TARGET_BITS` (2.0), `PIN_LEVEL` (8), `PIN_IN_BUDGET=1` to charge the pins to the
budget, `LEVELS` (3,4,5,6,7,8,9), `PIN` to allocate with a different pin file, `TEMPLATE`/`OUT` for step 7.

## Evaluation data

* **GPQA-Diamond** is a gated dataset (`Idavidrein/gpqa`). Accept its terms on the Hub and run `hf auth login`;
  `08_evaluate.sh` then rebuilds the 198 prompts from `assets/gpqa198_choice_order.json`, which stores only the order in
  which the four answers were displayed in the paper's runs (no question text). The prompts are byte-identical to the
  paper's.
* **LiveCodeBench** is graded with the official harness: the LiveCodeBench repository is cloned into `third_party/` at
  the commit the paper used, and the problem file for the 25.02-25.05 window is rebuilt from the official dataset.
* Sampling follows the Qwen model card (temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5), three samples per
  problem; an answer is credited only when the response stopped on its own inside the cap.

## Pin selection and held-out data

Pins are selected from MATH-500 traces only (`05_find_termination_experts.sh`), so GPQA-Diamond is never used for any
calibration or selection decision. The paper's reference pin set (223 experts) was the union over MATH-500 and
GPQA-Diamond traces; the MATH-500-only rule selects 175 of those 223 experts. See `docs/NOTES.md`.

## Layout

```
src/                 the pipeline (flat: modules import each other as top-level names, drivers run from src/)
  quantize/          the codebook quantizers (exact DP codebook, ternary path, KOTMS preprocessor)
scripts/             the drivers above, plus _common.sh (environment and the resumable `stage` helper)
assets/              gpqa198_choice_order.json
docs/NOTES.md        how this tree relates to the paper's runs, and what is still open
requirements.txt     tested versions in the header
```

## What this code does not do

* It does not fine-tune. Every stage is training-free and reads only forward and first-order information.
* It does not quantize anything but the routed experts. Attention, routers, shared experts, embeddings and the LM head
  stay BF16.
* It ships no checkpoints or data; everything is rebuilt from public sources with the scripts above.

## License

MIT (see `LICENSE`).
