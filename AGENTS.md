# AGENTS.md

Intel XPU AI stack (Arc B-series GPUs). Everything runs on `xpu`, never CUDA.
There is no CI, no test suite, and no linter/typecheck — verify by running the
scripts below.

## Two Python environments (don't mix them)

- **`.venv/`** — main env (declared in `pyproject.toml`, managed by `uv`).
  torch 2.12.0+xpu, diffusers 0.38.0, vllm-omni. ComfyUI runs from
  `vendor/ComfyUI` via `PYTHONPATH=vendor/ComfyUI` (not installed as a package).
- **`vendor/LTX-2/.venv/`** — separate env for LTX-2.5 video generation. Only
  this venv has `ltx_pipelines` and the model-checkpoint paths. It is built and
  XPU-patched by `setup.sh` only; never run `uv sync` or `uv pip install` inside
  `vendor/LTX-2` — its `pyproject.toml` sources torch from the cu132 index and
  would replace the XPU build with a CUDA one. Never install its `natten` extra
  (pins `torch==2.13.0+cu132`).

Always invoke python explicitly: `.venv/bin/python ...` or
`vendor/LTX-2/.venv/bin/python ...` (e.g. `ltx-gen.sh` uses the LTX venv,
`run-ltx-server.sh` uses the main venv).

## Setup

```bash
bash setup.sh          # clones vendor/ + builds both venvs + applies LTX XPU patches (idempotent)
rm -rf .venv && uv sync   # rebuild main env from scratch
```

`setup.sh` sources the Intel oneAPI toolchain (`/opt/intel/oneapi/setvars.sh`)
before `uv sync` — required to build vllm for XPU. The wrapper scripts
(`ltx-gen.sh`, `run-ltx-server.sh`) re-source it. If running raw
`.venv/bin/*` commands on a host where oneAPI is not global, source it first.

## Repo layout & gotchas

- `vendor/` is gitignored third-party checkouts, cloned by `setup.sh`. Prefer
  not to edit them; if you must, they will be re-cloned on a fresh setup. `setup.sh`
  applies LTX-2 XPU patches (`vendor/LTX-2/packages/*/src/...`): device selector,
  fp32 vocoder on XPU, and SDPA without cuDNN (which crashes with
  CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH).
- `pyproject.toml` is the source of truth for the main env: `triton` is
  `exclude-dependencies` (stock triton breaks XPU; import must resolve from
  `triton-xpu`); vllm/vllm-omni are `path` sources built with
  `no-build-isolation-package` + static `dependency-metadata` (their `setup.py`
  imports torch and cannot be executed during resolution).
- `output/*.mp4` is gitignored (generated videos).
- LTX-2.5 requires `num_frames = 8k + 1` (33, 121, 161, ...). `ltx_server.py`
  snaps input via `snap_frames_to_grid`.
- XPU quirks: native int4 matmul unsupported (emulate as two int8 GEMMs);
  SDPA falls back to MATH backend with a harmless warning; `xpu-smi`/Sysman
  can't read telemetry on the B70, so use `gpu_monitor.py` (xe sysfs).

## Commands

```bash
./ltx-gen.sh --distilled --prompt "A cat on a windowsill"   # fast 8-step, 33 frames
./ltx-gen.sh --prompt "..." --num-frames 121                # 50-step, ~10 min stage-1
./run-ltx-server.sh                                         # FastAPI demo, port 8090 (LTX_SERVER_PORT to override)
./gpu-monitor.sh --interval 2                               # sysfs GPU monitor (default device 0xe223 = B70)
.venv/bin/python bench_matmul.py --n 8192 --iters 10        # peak GEMM TFLOPS
```

ltx-gen.sh wraps `ltx-pipelines` CLIs with model paths from
`~/paul/models/ltx-2.5` (`LTX_MODELS` to override); defaults `--offload cpu
--quantization fp8-cast --seed 42`, 50-step mode also passes
`--num-inference-steps 50`. Any extra CLI flag is passed through.

Sanity check after setup:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.device_count())"  # 2.12.0+xpu 32
```
