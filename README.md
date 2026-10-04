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

## The engine/plug-in split (confirmed, de-risks the architecture above)

open-instruct is a generic, agent-agnostic distributed GRPO/DPO engine
(Ray orchestration, vLLM rollout generation, DeepSpeed/FSDP gradient
updates, DAPO/CISPO loss variants) — DR-Tulu is just one set of
plug-ins registered against its generic extension points, not a fork or
a bespoke trainer. This confirms the right scope of work here:

**Don't rebuild**: `grpo_fast.py`/`grpo.py` (the training loop itself),
`vllm_utils3.py` (rollout generation + weight sync), `rl_utils.py`/
`grpo_utils.py` (advantage/loss math). Multi-month distributed-systems
work, orthogonal to the actual research question, and not what this
thesis is about.

**Do build, as plug-ins against existing generic interfaces** — this is
the real, scoped, genuinely-original work for this repo:
- **Tool/environment**: likely close to zero new code —
  `environments/tools/generic_mcp.py` talks to *any* MCP server already,
  not just DR-Tulu's own. Point `--mcp_server_command` at mas-deepr's
  own `run_mcp_server.py` directly. **First real test to run** once
  both pieces are in place (de-risked by design, per the above, but
  tool *names* mas-deepr uses (`wikipedia_search`, etc.) vs. whatever
  a `VerifierFunction` expects still need checking — see below).
- **Reward — `VerifierFunction` subclass**: this is the one genuinely
  project-specific piece, and it's already half-built in `mas-deepr`:
  port `rl/rubric_reward.py`'s LLM-as-judge rubric-compliance scoring
  (+ the turn-count/hallucination penalties) into open-instruct's
  `VerifierFunction` interface (`ground_truth_utils.py`'s registry).
  This is real, original thesis work — not reused from DR-Tulu or
  anyone else.
- **Data/system prompts**: dataset schema + tool-calling system prompt
  are also genuinely project-specific — adapt `mas_deepr.data`'s
  benchmark loaders and `mas_deepr.prompts`' templates to whatever
  format open-instruct's data pipeline expects, rather than writing
  either from scratch.

## What's not done yet

`sft/llama-factory/` and `rl/open-instruct/` are empty placeholders —
to be cloned in directly (per the Eden plan's Phase 3) once on a
machine with real GPU access; not installed on a laptop. The MCP
tool-backend first-contact test (above) and the `VerifierFunction`
reward port are the two concrete next pieces of real work.

See `mas-deepr`'s `docs/plan/team-task-split-2026-10.md` (local-only,
not in git) for the fuller context/task breakdown this repo exists to
execute.
