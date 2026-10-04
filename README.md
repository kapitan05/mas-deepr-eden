# mas-deepr-eden

Full-scale RL/SFT training infrastructure for the mas-deepr thesis
project, built on the WUT Eden SLURM cluster — a separate track from
the [mas-deepr](https://github.com/kapitan05/mas-deepr) repo's
OpenPipe-ART-based pipeline.

## Why a separate repo

mas-deepr's existing stack (`agent_framework`/MAF topology, DSPy
compilation, OpenPipe ART for GRPO/DPO) runs fine at the scale W&B's
serverless infra allows, but doesn't scale to real multi-GPU cluster
training. This repo adopts production RL/SFT infrastructure instead:

- **`sft/`** — [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
  for supervised fine-tuning (T3S cold-start equivalent, full-parameter
  at 8B/14B).
- **`rl/`** — [open-instruct](https://github.com/allenai/open-instruct)
  (the same training code DR-Tulu itself uses) for GRPO/DPO at scale.
- **`agent/`** — DR-Tulu's own agent library, as the rollout-time tool
  backend open-instruct's RL loop expects.

Each of the above is vendored as its own subdirectory with its **own**
independent `uv` environment — their dependency pins (`deepspeed==0.15.4`,
cu124 torch wheels, `flash-attn`, Ray) are specific and would conflict
with mas-deepr's own `rl-art` extra (and with each other) if forced into
one shared environment.

## Reusing mas-deepr: dependency, not a copy

The agentic environment that already works — MAF topology, MCP tool
backend (wikipedia/semantic-scholar/pubmed search with tuned rate
limits), benchmark loaders (FRAMES/ResearchQA/HealthBench/MuSiQue/
HotpotQA), the LLM-as-judge eval harness — is **not duplicated here**.
This repo's own orchestration/glue code depends on `mas-deepr` as an
installed package:

```bash
uv add "mas-deepr @ git+https://github.com/kapitan05/mas-deepr.git"
```

Pull Oleksii's ongoing improvements to `mas-deepr`'s `main` on demand:

```bash
uv lock --upgrade-package mas-deepr
```

Installed **without** the `rl-art` extra — `torch`/`trl`/`peft`/
`openpipe-art` never enter this repo's base environment (confirmed safe:
`mas_deepr.rl`'s own lazy-loading design means importing the package
doesn't eagerly pull in the heavy RL stack unless `mas_deepr.rl.*` is
reached into directly).

## What's not done yet

`sft/llama-factory/` and `rl/open-instruct/` are empty placeholders —
to be cloned in directly (per the Eden plan's Phase 3) once on a
machine with real GPU access; not installed on a laptop. The MCP
tool-backend compatibility between mas-deepr's server and open-instruct's
`--mcp_server_command` flag is unverified — first real test to run once
both pieces are in place.

See `mas-deepr`'s `docs/plan/team-task-split-2026-10.md` (local-only,
not in git) for the fuller context/task breakdown this repo exists to
execute.
