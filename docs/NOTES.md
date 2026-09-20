# Notes for maintainers

## How this tree relates to the paper

The paper's Qwen3.6-35B-A3B checkpoints were produced by exactly the modules in `src/` (their import closure from the
drivers is the whole directory). The drivers in `scripts/` are re-issued for release: each one is a resumable chain of
the same commands, with every machine-specific path replaced by `RTAQ_ROOT`, `MODEL_DIR`, `PYTHON` and `GPUS`.

Two things differ from the reference run and are deliberate:

1. **Pin sources.** The paper's reference pin set (223 experts) was the union over MATH-500 *and* GPQA-Diamond traces of
   the BF16 and unpinned 2-bit models. `05_find_termination_experts.sh` uses MATH-500 only, so GPQA-Diamond is held
   out from every calibration and selection decision. On the same probe outputs the MATH-500-only rule selects 175 of
   those 223 experts (all 175 are in the reference set). To reproduce the reference set exactly, probe the two GPQA
   traces as well and pass all four `.npz` files to `build_pins.py`.
2. **Evaluation data.** The GPQA-Diamond prompts are rebuilt from the gated dataset with the shipped choice order
   (`assets/gpqa198_choice_order.json`); the LiveCodeBench problem file is rebuilt from the official dataset at the
   paper's date window. Both are byte-identical to what the paper's runs used.

## Verified before release

* every module compiles; no absolute paths, hostnames, tokens or personal identifiers in `src/`, `scripts/`, `assets/`
* `build_pins.py` applied to the paper's four probe outputs reproduces the 223-expert reference set exactly
* `prepare_gpqa_docs.py` reproduces the 198 cached prompts byte for byte and the same gold letters
* Stage 3 `verify_gptq_gamma.py` passes on the reference checkpoint (weights on the allocated codebook grid, non-expert
  tensors untouched)

## Still open

* **Tests** - the research tree's unit tests for the codebook quantizer and the branch-and-bound search have not been
  ported; they would live in `tests/`.
* `probe_termination_routing.py` imports a private `transformers` module path (see `requirements.txt`).
* The corpus builder streams `nvidia/Nemotron-SFT-Math-v3` from the Hub at build time (its single file is 154 GB).
