# FlashInfer-Bench MLSys 2026 - AI Agent Guide

## What We Know For Certain

### The Contest
- **Goal:** Create GPU kernels for DeepSeek-V3 FP8 MoE that are faster than the reference
- **Hardware:** NVIDIA B200 (access via Modal)
- **Current Baseline:** Reference implementation runs at ~14.6ms on B200
- **Dataset:** `flashinfer-ai/mlsys26-contest` on HuggingFace

### Files in This Repo

| File | What It Is |
|------|-----------|
| `AGENT_GUIDE.md` | This file |
| `modal_flashinfer_full.py` | Modal script for B200 benchmarking |
| `solution_moe_reference.json` | The correct reference implementation (extracted from contest definition) |
| `flashinfer_trace/` | Original repo data (git restored) |
| `mlsys26_contest/` | Contest dataset (downloaded locally) |

### How to Run

```bash
# 1. Setup dataset on Modal Volume (one time)
modal run modal_flashinfer_full.py::setup

# 2. Test reference (verifies everything works)
modal run modal_flashinfer_full.py::main --solution solution_moe_reference.json
# Expected: Status=PASSED, Latency~14.6ms, Speedup=1.00x

# 3. Create your solution
cp solution_moe_reference.json my_solution.json
# Edit the "content" field in sources[0] to add your code

# 4. Benchmark your solution
modal run modal_flashinfer_full.py::main --solution my_solution.json
```

### Contest Tracks

| Track | Definition Name |
|-------|----------------|
| Track A (MoE) | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` |
| Track B (DSA) | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| Track B (Indexer) | `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` |
| Track C (GDN) | `gdn_decode_qk4_v8_d128_k_last` |
| Track C (GDN Prefill) | `gdn_prefill_qk4_v8_d128_k_last` |

### Solution Format

A solution is JSON with this structure:

```json
{
  "name": "your_solution_name",
  "definition": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
  "description": "Description",
  "author": "your-team",
  "spec": {
    "language": "python",
    "target_hardware": ["NVIDIA_B200"],
    "dependencies": ["torch >= 2.5"],
    "entry_point": "main.py::run"
  },
  "sources": [
    {
      "path": "main.py",
      "content": "import torch\ndef run(...): ..."
    }
  ]
}
```

Requirements:
- Must have `def run(...)` with exact signature from definition
- Must return correct output (numerically matches reference within tolerance)
- Entry point must be `main.py::run`

### The Reference Implementation

The file `solution_moe_reference.json` contains the exact reference code from the contest definition. It:
- Passes all correctness checks
- Takes ~14.6ms on B200
- Is the baseline you must beat

You can read its code to understand:
- The exact FP8 block-scale dequantization formula
- The DeepSeek-V3 routing logic
- The expected input/output formats

### What Constitutes Success

A successful solution must:
1. Pass correctness check (max error < tolerance)
2. Have latency < reference (~14.6ms for MoE)
3. Show Speedup > 1.00x

### What I Don't Know (Don't Trust Me On These)

I do NOT know:
- What specific optimizations will work
- What Triton kernel configurations are best
- How to write a correct Triton kernel for this (my attempts had bugs)
- What the optimal memory layout is
- How to fuse operations correctly

**You are on your own for optimization strategies.** The reference works. Figure out how to make it faster while keeping it correct.

### Debugging

If your solution fails:

```bash
# Check the error message from the benchmark
# Common issues:
# - Numerical mismatch (wrong FP8 dequantization)
# - Shape errors
# - CUDA errors in Triton kernels
```

To iterate locally without Modal:
```bash
# Use the test_moe_solution.py pattern (but I deleted it, recreate if needed)
# Or just keep testing with Modal - you have unlimited B200 credits
```

### Resources

- Contest definition (read the JSON): `mlsys26_contest/definitions/moe/*.json`
- Reference code: `solution_moe_reference.json`
- Contest repo: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest

### Notes

1. Start from `solution_moe_reference.json` - it's the only guaranteed correct implementation
2. Test on B200 early and often using Modal
3. Verify correctness before optimizing for speed
4. The contest evaluates on bare-metal B200, not Modal (but Modal gives you good estimates)

That's it. No more advice from me - I was giving wrong Triton tips earlier. Good luck.
