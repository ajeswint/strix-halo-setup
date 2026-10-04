# Working rules for agents in this repo

- Read `README.md` first; it is the current state of the machine and the decision log. Files under `bench/`, `etc/`, `systemd/`, `envs/`, `models/`, `scripts/` are copies of live files; the tree mirrors their real paths.
- Terse. Mechanism-level explanations. Yes/no first when the question is yes/no.
- Exact copy-paste commands only. Never `<placeholders>` in command blocks.
- Verify against current upstream docs before asserting versions or install steps. Cite the URL.
- Measured numbers beat claims. If you propose a tuning change, say what benchmark delta would justify keeping it.
- Simplest fix that is set-and-forget. No polling loops, watchdogs or demo self-tests unless there is evidence a setting reverts.
- Do not propose: Docker (podman is installed), Postgres (SQLite decided), Unsloth in the torch env, `download.pytorch.org` ROCm wheels, IOMMU off, GTT above 108 GiB.
- When installing into `/data/envs/torch` with the AMD wheel index, always add `--index-strategy unsafe-best-match` and the constraints file `envs/torch-constraints.txt`.
- Own errors plainly and correct forward.
