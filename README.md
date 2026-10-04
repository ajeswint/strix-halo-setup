# strix-halo-setup

Configuration, benchmarks and decisions for a Framework Desktop (Ryzen AI Max+ 395 / Strix Halo, Radeon 8060S gfx1151, 128 GB unified LPDDR5X) used as a local LLM box. Everything here is a copy of the live file on the machine; paths in the tree mirror where the file lives.

## Hardware

- Framework Desktop, Ryzen AI Max+ 395, Radeon 8060S iGPU (gfx1151), 128 GB LPDDR5X-8000 (256 GB/s theoretical)
- OS: WD_Black SN7100 1 TB in the back M.2 slot (enumerates as `nvme1n1`)
- Data: WD_Black SN7100X 2 TB in the front slot, 4K LBA, XFS at `/data` (`scripts/data-drive-setup.sh`). Address disks by `/dev/disk/by-id`, never `nvmeXn1`.
- BIOS 3.06, EC and PD firmware current (2026-10-03). BIOS update produced no performance change.

## Software state (verified 2026-10-04)

| Layer | State |
|---|---|
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, in-tree `amdgpu` |
| System ROCm | 10.0.0 (`amdrocm-core10.0-gfx1151`), `scripts/rocm-install.sh`, `scripts/rocm-verify.sh` |
| GTT | tuned to 108 GiB visible to the GPU |
| Power | `tuned` + `tuned-ppd`, profile `accelerator-performance`, persisted via `etc/tuned/` (replaced power-profiles-daemon) |
| Cooling | `systemd/fan-peg.service`: `framework_tool --fansetduty 100` at boot and after resume. See *Why the fan is pegged*. |
| llama.cpp | `/data/src/llama.cpp` HEAD 836d57176, `build-hip` (gfx1151) and `build-vk`; `etc/ld.so.conf.d/rocm.conf` puts the HIP libs on the linker path |
| Serving | `systemd/llama-deepseek.service`: DeepSeek V4 Flash UD-IQ2_M, HIP, 64k context, `127.0.0.1:8080`, alias `deepseek-v4-flash`. Started manually, not enabled at boot (pins ~85 GiB GTT). |
| Models | `/data/models`: Qwen3.6-35B-A3B UD-Q4_K_XL, DeepSeek V4 Flash 0731 UD-IQ2_M (3 shards). `models/fetch.sh` + `systemd/model-download.service` (disabled, download complete). |
| Python / training | `/data/envs/torch`: uv venv on system Python 3.14.4. torch 2.13.0+rocm10.0.0 via `torch[device-gfx1151]` from `stable.repo.amd.com/rocm/whl-next/` (bundles its own ROCm user-space runtime; nothing loads from `/opt/rocm`), transformers 5.18.0, peft 0.21.2, trl 1.14.1, datasets 5.0.1, accelerate 1.15.0. `envs/install-torch-env.sh` reproduces it; `envs/freeze.txt` is the exact result. `HF_HOME=/data/hf-cache`. |
| Tooling | uv, gh, git-lfs, podman (rootless) + distrobox, sqlite3, tmux, btop, nvtop, aria2, fd-find |
| Cursor | AppArmor adjustments in `etc/apparmor.d/` so Cursor's terminal sandbox runs under Ubuntu's user-namespace restriction. Keep the local conffile on Cursor updates. |

Not installed, by decision: Docker (podman instead), Postgres (SQLite, see below), Unsloth (would need Python 3.10-3.13 and its own torch; separate env if ever), bitsandbytes.

## Why the fan is pegged

The EC's automatic fan curve drives fan header 0, which has no fan on it (reads 0 RPM at 100% duty). The CPU heatsink fan is on header 1, which the curve never ramps. Under sustained load the APU hit ~100 °C in two minutes and fell off boost to ~105 W. With both headers forced to 100% (`fan-peg.service`) the chip holds 138 W / ~2885 MHz for ~8 minutes, then 120 W / ~2790 MHz at 68-71 °C indefinitely — Framework's published spec. Prompt processing rose 35-40%. Hottest point in the whole suite was 84 °C (64k-context HIP run). Open: check the silkscreen at the fan plug; if it says "APU Fan", the header mapping is an EC firmware bug to report to Framework.

## Benchmarks

`bench/bench.sh <label> <gguf>` runs llama-bench under Framework's conditions (pp2048 / tg128, `-ngl 999 -fa on -lm none`) on HIP and Vulkan and appends to `bench/results.jsonl`; `bench/report.py` summarises. `bench/fanpeg-suite.sh` is the cooling A/B suite; `bench/fanpeg-mon.log` is its 2 s power/clock/temp trace. `bench/post-reboot.sh` checks BIOS, cmdline, GTT, power profile after a reboot, then benches.

Fan-pegged results (t/s):

| Model | HIP pp / tg | Vulkan pp / tg |
|---|---|---|
| Qwen3.6-35B-A3B Q4_K_XL, depth 0 | 1392 / 53.1 | 1390 / 61.7 |
| DeepSeek V4 Flash IQ2_M, depth 0 | 261 / 16.4 | 223 / 18.6 |
| DeepSeek, depth 8k | 202 / 15.0 | 174 / 17.7 |
| DeepSeek, depth 32k | 119 / 13.5 | 108 / 13.6 |
| DeepSeek, depth 64k | — / 11.7 | — / 11.1 |

Token generation is memory-bandwidth bound (~117 GB/s effective); power and clock knobs do not move it. HIP wins prompt processing at every depth; Vulkan wins generation at short context; they meet around 32k. HIP is used for long-context serving.

PyTorch on gfx1151, bf16 square matmul, warm, TFLOPS: 2048² 34.7, 4096² 36.4, 8192² 32.3 (hipBLASLt, which torch selects by default). LoRA smoke test (random-init 100M Llama, r=16, bf16): loss decreases, finite, 14.4k tok/s, 2 GiB peak.

## Decisions

- **Measure, don't assume.** The machine follows Framework's published local-AI setup as baseline; deviations need a benchmark delta. Dropped after measurement or analysis: IOMMU off (GPU and NPU are already identity-mapped; would cost USB4/NVMe DMA protection), GTT above 108 GiB (64k DeepSeek peaks at 86 GiB), GPU DPM forced high (+15-20 W idle for ~2%).
- **Set-and-forget.** Settings that must persist are applied once at the points where they can reset (boot, resume). No polling loops, watchdogs or self-tests unless observed behaviour shows a revert.
- **SQLite, one writer**, WAL mode, for the orchestrator's memory. Postgres + pgvector only when parallel writers hit lock errors or maintained migrations are needed. Vectors later via sqlite-vec in the same file.
- **PyTorch from AMD's gfx1151 wheels**, not `download.pytorch.org` ROCm 7.x wheels (no gfx1151 kernels; Unsloth's AMD page assumes an MI300 via `HSA_OVERRIDE_GFX_VERSION`).
- **uv gotcha:** the AMD wheel index also hosts `fsspec` (one version). With uv's default first-index strategy that pins `fsspec` and silently backtracks `datasets`/`trl` to 2020-era releases. Always pass `--index-strategy unsafe-best-match` when installing into the torch env with that index.
- **DeepSeek V4 Flash is a background reviewer**, not an interactive model (1-3 min per turn cached, 4-6 uncached). Interactive and agentic work uses lighter MoE models (Qwen3.6-35B-A3B).

## Roadmap

1. Orchestrator + SQLite memory: structured verdicts (finding, file, severity, accepted/rejected + reason) retrieved into prompts. DeepSeek as an hourly adversarial reviewer of diffs; static system/rules/repo summary first in the prompt, changing material last, for cache hits.
2. Small-model loop: Qwen3.6-35B-A3B made strong via tools, tests and best-of-N, with DeepSeek as judge. Tests and compiler outrank the judge.
3. One timed LoRA job, kept only if it beats the base model on held-out tasks. Verdicts are not training pairs; a dataset has to be built. Risks: model collapse, bad judge.
4. DaVinci Resolve Studio in a container (pending: licence type, distrobox vs davincibox).
5. Server-class hardware research, criterion bandwidth per dollar with confirmed software support.
