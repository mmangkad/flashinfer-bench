"""
Modal script for full FlashInfer-Bench with dedicated Volume for dataset.

Usage:
    # Setup: Download dataset to Volume (run once)
    modal run modal_flashinfer_full.py::setup

    # Run benchmark (uses cached dataset)
    modal run modal_flashinfer_full.py --solution solution_moe_working_v2.json

    # Full competition benchmark
    modal run modal_flashinfer_full.py --solution solution_moe_working_v2.json --competition
"""

import json
from pathlib import Path

import modal

app = modal.App("flashinfer-full")

# Configuration
GPU_CONFIG = "b200"
VOLUME_NAME = "mlsys26-contest-dataset"
DATASET_PATH = "/dataset"
# Contest dataset uses FIB_DATASET_PATH env var
CONTEST_DATASET_PATH = "/dataset/mlsys26-contest"

# Standalone Python 3.12
standalone_python_url = "https://github.com/astral-sh/python-build-standalone/releases/download/20260203/cpython-3.12.12+20260203-x86_64_v4-unknown-linux-gnu-pgo+lto-full.tar.zst"

base_setup_commands = [
    "RUN apt update && apt list --upgradable 2>/dev/null | grep -v '^Listing' | awk -F/ '{print $1}' | xargs -r apt install -y --allow-change-held-packages && apt full-upgrade -y --allow-change-held-packages && apt install -y wget zstd && rm -rf /var/lib/apt/lists/* && apt clean",
    f"RUN wget -O /tmp/python.tar.zst {standalone_python_url}",
    "RUN cd /tmp && tar -I zstd -xf python.tar.zst",
    "RUN cp -r /tmp/python/install/* /usr/local/",
    "ENV TERMINFO_DIRS=/etc/terminfo:/lib/terminfo:/usr/share/terminfo:/usr/lib/terminfo",
    "RUN rm -rf /tmp/python.tar.zst /tmp/python",
    "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/",
]

# Create image
image = (
    modal.Image.from_registry("nvcr.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04")
    .entrypoint([])
    .dockerfile_commands(base_setup_commands)
    .run_commands(
        "uv pip install --python /usr/local/bin/python3.12 --compile-bytecode --upgrade pip",
        "uv pip install --python /usr/local/bin/python3.12 --compile-bytecode torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",
        "uv pip install --python /usr/local/bin/python3.12 --compile-bytecode triton numpy safetensors pydantic",
        "uv pip install --python /usr/local/bin/python3.12 --compile-bytecode flashinfer-bench --no-deps",
        "uv pip install --python /usr/local/bin/python3.12 --compile-bytecode click tabulate tqdm packaging requests einops ninja apache-tvm-ffi",
        "uv pip install --python /usr/local/bin/python3.12 --compile-bytecode flashinfer-python --no-deps",
        "uv pip install --python /usr/local/bin/python3.12 --compile-bytecode huggingface-hub",
        # Install git after uv pip installs
        "apt update && apt install -y git && rm -rf /var/lib/apt/lists/* && apt clean",
    )
)

# Create or get volume
dataset_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def print_system_info():
    """Print GPU info."""
    import subprocess

    import torch

    print("=" * 70)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"GPU: {result.stdout.strip()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print("=" * 70)


@app.function(
    image=image, volumes={DATASET_PATH: dataset_volume}, timeout=1800  # 30 minutes for download
)
def download_dataset(force: bool = False) -> dict:
    """
    Download MLSys26 contest dataset to Modal Volume.
    Run this once, then reuse for all benchmarks.
    """
    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download

    dataset_dir = Path(CONTEST_DATASET_PATH)

    # Check if already exists
    if dataset_dir.exists() and not force:
        definitions = list((dataset_dir / "definitions").rglob("*.json"))
        print(f"Dataset already exists with {len(definitions)} definitions")
        return {"status": "exists", "path": str(dataset_dir), "definitions": len(definitions)}

    print("=" * 70)
    print("Downloading MLSys26 Contest Dataset")
    print("=" * 70)
    print("Repo: flashinfer-ai/mlsys26-contest")
    print("This may take 5-10 minutes...")
    print("=" * 70)

    try:
        snapshot_download(
            repo_id="flashinfer-ai/mlsys26-contest",
            repo_type="dataset",
            local_dir=str(dataset_dir),
            local_dir_use_symlinks=False,
            max_workers=256,
        )

        # Count what we got
        definitions = list((dataset_dir / "definitions").rglob("*.json"))
        workloads = list((dataset_dir / "workloads").rglob("*.jsonl"))

        result = {
            "status": "downloaded",
            "path": str(dataset_dir),
            "definitions": len(definitions),
            "workloads": len(workloads),
        }

        print(f"\n✓ Download complete!")
        print(f"  Definitions: {len(definitions)}")
        print(f"  Workloads: {len(workloads)}")

        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.function(gpu=GPU_CONFIG, image=image, volumes={DATASET_PATH: dataset_volume}, timeout=600)
def run_full_benchmark(
    solution_content: dict,
    definition_name: str,
    iterations: int = 50,
    warmup_runs: int = 10,
    save_results: bool = True,
) -> dict:
    """
    Run full benchmark with real workload data from Volume.
    """
    import os
    import sys
    import time
    from pathlib import Path

    import torch

    # Add flashinfer_bench to path
    sys.path.insert(0, "/usr/local/lib/python3.12/site-packages")

    # Set environment variable for contest dataset
    os.environ["FIB_DATASET_PATH"] = CONTEST_DATASET_PATH

    print_system_info()

    dataset_dir = Path(CONTEST_DATASET_PATH)

    # Check dataset exists
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}")
        print("Run: modal run modal_flashinfer_full.py::setup")
        return {"error": "Dataset not found", "status": "failed"}

    print(f"\n📁 Contest Dataset: {dataset_dir}")
    print(f"   FIB_DATASET_PATH={CONTEST_DATASET_PATH}")

    # Save solution
    solution_name = solution_content.get("name", "unknown")
    solution_dir = dataset_dir / "solutions" / definition_name
    solution_dir.mkdir(parents=True, exist_ok=True)

    solution_file = solution_dir / f"{solution_name}.json"
    with open(solution_file, "w") as f:
        json.dump(solution_content, f)

    print(f"💾 Solution saved: {solution_file}")

    # Import here to avoid early import issues
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    print(f"\n📊 Loading trace set...")
    try:
        trace_set = TraceSet.from_path(str(dataset_dir))
        print(f"✓ Loaded {len(trace_set.definitions)} definitions")
        print(f"✓ Loaded {len(trace_set.workloads)} workload sets")
    except Exception as e:
        print(f"Error loading trace set: {e}")
        return {"error": str(e), "status": "failed"}

    if definition_name not in trace_set.definitions:
        available = list(trace_set.definitions.keys())[:5]
        print(f"Definition '{definition_name}' not found!")
        print(f"Available: {available}...")
        return {"error": f"Definition not found: {definition_name}"}

    # Check workloads exist
    if definition_name not in trace_set.workloads or not trace_set.workloads[definition_name]:
        print(f"⚠️ No workloads found for {definition_name}")
        print("Using synthetic test...")
        # Fallback to synthetic
        return run_synthetic_benchmark(solution_content, definition_name, iterations)

    print(f"\n🎯 Definition: {definition_name}")
    print(f"🔧 Solution: {solution_name}")
    print(f"📈 Iterations: {iterations}")
    print(f"🔥 Warmup: {warmup_runs}")

    # Configure benchmark
    config = BenchmarkConfig(
        warmup_runs=warmup_runs,
        iterations=iterations,
        num_trials=3,
        rtol=1e-2,
        atol=1e-2,
        use_isolated_runner=False,
        definitions=[definition_name],
        solutions=[solution_name],
        timeout_seconds=300,
    )

    print("\n" + "=" * 70)
    print("🏁 STARTING BENCHMARK")
    print("=" * 70)

    start_time = time.time()

    try:
        benchmark = Benchmark(trace_set, config)
        benchmark.run_all()

        results = {
            "status": "success",
            "definition": definition_name,
            "solution": solution_name,
            "duration_seconds": time.time() - start_time,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        }

        # Extract and print results
        if definition_name in trace_set.traces:
            traces = trace_set.traces[definition_name]
            print(f"\n📊 Generated {len(traces)} traces")

            for trace in traces:
                print(f"\n✓ {trace.solution}")
                if trace.evaluation:
                    # Convert Pydantic model to dict
                    eval_dict = (
                        trace.evaluation.model_dump()
                        if hasattr(trace.evaluation, "model_dump")
                        else dict(trace.evaluation)
                    )
                    status = eval_dict.get("status", "unknown")
                    print(f"  Status: {status}")

                    perf = eval_dict.get("performance")
                    if perf:
                        latency = perf.get("latency_ms", "N/A")
                        speedup = perf.get("speedup_factor", "N/A")
                        print(
                            f"  Latency: {latency:.3f} ms"
                            if isinstance(latency, float)
                            else f"  Latency: {latency}"
                        )
                        print(
                            f"  Speedup: {speedup:.2f}x"
                            if isinstance(speedup, float)
                            else f"  Speedup: {speedup}"
                        )
                    else:
                        print(f"  Latency: N/A (correctness check failed)")

                    corr = eval_dict.get("correctness")
                    if corr:
                        max_err = corr.get("max_absolute_error", "N/A")
                        print(
                            f"  Max error: {max_err:.2e}"
                            if isinstance(max_err, float)
                            else f"  Max error: {max_err}"
                        )

        # benchmark.close()  # Not needed in this version

        print("\n" + "=" * 70)
        print("✅ BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"Duration: {results['duration_seconds']:.2f}s")

        return results

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


def run_synthetic_benchmark(solution_content: dict, definition_name: str, iterations: int) -> dict:
    """Fallback synthetic benchmark if no workloads."""
    import time

    import torch

    print("\n" + "=" * 70)
    print("🧪 SYNTHETIC BENCHMARK (No workloads found)")
    print("=" * 70)

    # Extract and run solution
    sources = solution_content.get("sources", [])
    if not sources:
        return {"error": "No source code"}

    main_py = sources[0].get("content", "")

    # Write to file
    import importlib.util
    import os
    import sys

    temp_dir = "/tmp/solution"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "main.py")

    with open(temp_file, "w") as f:
        f.write(main_py)

    if temp_dir not in sys.path:
        sys.path.insert(0, temp_dir)

    if "main" in sys.modules:
        del sys.modules["main"]

    spec = importlib.util.spec_from_file_location("main", temp_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["main"] = module
    spec.loader.exec_module(module)

    run_fn = getattr(module, "run", None)
    if run_fn is None:
        return {"error": "No run function"}

    # Generate test data for MoE
    if "moe" in definition_name:
        H, I, E_GLOBAL, E_LOCAL, TOP_K = 7168, 2048, 256, 32, 8
        G1_OUT = 4096
        BLOCK = 128
        device = "cuda"
        seq_len = 8

        torch.manual_seed(42)
        routing_logits = torch.randn(seq_len, E_GLOBAL, dtype=torch.float32, device=device)
        routing_bias = torch.zeros(E_GLOBAL, dtype=torch.bfloat16, device=device)
        hidden_states = torch.randn(seq_len, H, dtype=torch.float32, device=device).to(
            torch.float8_e4m3fn
        )
        hidden_states_scale = (
            torch.randn(H // BLOCK, seq_len, dtype=torch.float32, device=device).abs() * 0.01
        )
        gemm1_weights = torch.randn(E_LOCAL, G1_OUT, H, dtype=torch.float32, device=device).to(
            torch.float8_e4m3fn
        )
        gemm1_weights_scale = (
            torch.randn(
                E_LOCAL, G1_OUT // BLOCK, H // BLOCK, dtype=torch.float32, device=device
            ).abs()
            * 0.01
        )
        gemm2_weights = torch.randn(E_LOCAL, H, I, dtype=torch.float32, device=device).to(
            torch.float8_e4m3fn
        )
        gemm2_weights_scale = (
            torch.randn(E_LOCAL, H // BLOCK, I // BLOCK, dtype=torch.float32, device=device).abs()
            * 0.01
        )

        # Warmup
        for _ in range(3):
            _ = run_fn(
                routing_logits,
                routing_bias,
                hidden_states,
                hidden_states_scale,
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                0,
                2.5,
            )
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = run_fn(
                routing_logits,
                routing_bias,
                hidden_states,
                hidden_states_scale,
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                0,
                2.5,
            )
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        mean_time = sum(times) / len(times)

        print(f"\n📊 SYNTHETIC RESULTS")
        print(f"  Latency: {mean_time:.3f} ms")
        print(f"  Throughput: {seq_len / (mean_time / 1000):.1f} tokens/sec")

        return {
            "status": "success (synthetic)",
            "latency_ms": mean_time,
            "note": "No real workloads found, used synthetic data",
        }

    return {"error": "Synthetic benchmark not implemented for this definition"}


# ==============================================================================
# Local Entrypoints
# ==============================================================================


@app.local_entrypoint()
def setup(force: bool = False):
    """
    Setup: Download dataset to Modal Volume (run once).
    """
    print("Setting up flashinfer-trace dataset...")
    result = download_dataset.remote(force=force)

    if result["status"] == "exists":
        print(f"\n✓ Dataset already exists!")
        print(f"  Definitions: {result['definitions']}")
    elif result["status"] == "downloaded":
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"  Definitions: {result['definitions']}")
        print(f"  Workloads: {result['workloads']}")
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")


@app.local_entrypoint()
def main(
    solution: str = None,
    definition: str = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    competition: bool = False,
    iterations: int = 10,
    warmup: int = 5,
):
    """
    Run full benchmark on B200 with dataset from Volume.

    Args:
        solution: Path to solution JSON file
        definition: Definition name to benchmark
        competition: Use competition settings (50 iterations)
        iterations: Number of iterations (ignored if competition=True)
        warmup: Warmup iterations
    """
    if solution is None:
        print("Usage:")
        print("  modal run modal_flashinfer_full.py --solution my_solution.json")
        print("\nFirst time setup:")
        print("  modal run modal_flashinfer_full.py::setup")
        return

    solution_path = Path(solution)
    if not solution_path.exists():
        print(f"ERROR: Solution file not found: {solution_path}")
        return

    with open(solution_path) as f:
        solution_content = json.load(f)

    # Competition settings
    if competition:
        iterations = 50
        warmup = 10
        print("🏆 COMPETITION MODE")

    print(f"\n🚀 Running benchmark on B200...")
    print(f"   Solution: {solution_content.get('name')}")
    print(f"   Definition: {definition}")
    print(f"   Iterations: {iterations}")

    result = run_full_benchmark.remote(
        solution_content=solution_content,
        definition_name=definition,
        iterations=iterations,
        warmup_runs=warmup,
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if result.get("status", "").startswith("success"):
        print("✅ SUCCESS")
        print(f"   GPU: {result.get('gpu', 'unknown')}")
        print(f"   Duration: {result.get('duration_seconds', 'N/A'):.2f}s")
    else:
        print("❌ FAILED")
        print(f"   Error: {result.get('error', 'Unknown')}")

    print("=" * 70)


@app.local_entrypoint()
def status():
    """Check dataset status in Volume."""
    result = download_dataset.remote()
    print(f"\nDataset status: {result['status']}")
    if "definitions" in result:
        print(f"Definitions: {result['definitions']}")
    if "workloads" in result:
        print(f"Workloads: {result['workloads']}")
