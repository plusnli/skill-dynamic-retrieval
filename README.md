# Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval

[![arXiv](https://img.shields.io/badge/arXiv-2606.04391-b31b1b.svg)](https://arxiv.org/pdf/2606.04391)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

Official implementation of [**Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval**](https://arxiv.org/pdf/2606.04391).

This repository provides **State-Grounded Dynamic Retrieval (SGDR)**, an online skill-learning method for WebArena-style web agents, together with retained baseline runners and evaluation utilities.

SGDR maintains a growing JSONL library of reusable skills. During task solving, the agent summarizes the current browser state, retrieves skills that are relevant to both the task goal and the current state, and dynamically injects the selected skills into the action space. After a successful trajectory, SGDR cleans the trace, identifies reusable action windows, synthesizes new skills, and appends them to the skill library for future tasks.

## Paper

- Paper: [arXiv:2606.04391](https://arxiv.org/pdf/2606.04391)

## Repository Layout

```text
skill-dynamic-retrieval/
  browsergym/          BrowserGym/WebArena dependencies used by the project
  sgdr/                SGDR agent, retrieval, induction, and evaluation code
    actions/           Base action sets and learned skill libraries
    autoeval/          Trajectory evaluation utilities
    config_files/      WebArena task configuration template and generator
    induce/            Skill induction pipelines
    retrieval/         State summarization, embedding, and skill retrieval
    workflows/         Workflow-memory files for AWM-style baselines
```

Most project commands should be run from `skill-dynamic-retrieval/sgdr/`.

## Setup

Create a Python environment and install the main dependencies:

```bash
conda create -n sgdr python=3.10
conda activate sgdr

cd skill-dynamic-retrieval/sgdr
pip install browsergym==0.10.2 browsergym-webarena==0.10.2
pip install -r requirements.txt
pip install gymnasium playwright==1.49.0 litellm
playwright install chromium
```

SGDR expects a running WebArena deployment. Configure its public host locally:

```bash
cp host.local.example.sh host.local.sh
# Edit host.local.sh and set WEBARENA_HOST to your WebArena host.
```

`host.local.sh` is ignored by Git because it is machine-specific. Generated task configs under `config_files/*.json` are also ignored because they embed local service URLs.

Set your OpenAI-compatible API key in the shell or in a local ignored `.env` file:

```bash
export OPENAI_API_KEY="your-api-key"
```

Then load the runtime environment and generate WebArena task configs:

```bash
source env.sh
python config_files/generate_test_data.py
```

## Quick Start

Run one BrowserGym/WebArena task with the default agent:

```bash
cd skill-dynamic-retrieval/sgdr
source env.sh

python run_demo.py \
  --task_name webarena.21 \
  --websites shopping \
  --headless
```

Run SGDR online over a task range:

```bash
python run_online.py \
  --experiment sgdr \
  --website shopping \
  --task_ids "21-25" \
  --model gpt-4.1
```

The retained experiment choices are:

```text
sgdr        State-grounded dynamic retrieval
awm         Workflow-memory baseline
asi         Action-skill induction baseline
cer_online  CER-style online experience retrieval baseline
```

Allowed websites are `shopping`, `admin`, `reddit`, `gitlab`, and `map`.

## SGDR Pipeline

For each task, `run_online.py --experiment sgdr` performs:

1. **Solve**: run the web agent with dynamic skill retrieval.
2. **Evaluate**: judge whether the trajectory completed the task.
3. **Clean**: remove invalid or unusable steps.
4. **Induce**: synthesize reusable skills from successful cleaned trajectories.
5. **Update**: append new skills to the JSONL skill library.

The skill library is stored under:

```text
sgdr/actions/_skill_lib/sgdr_{model}/{website}.jsonl
```

At the start of a new SGDR run, an existing library for the same model and website is archived under `_history/` unless `--reuse_skill_lib` is passed.

## Useful Commands

Evaluate completed runs:

```bash
python eval_results.py --experiment sgdr --model gpt-4.1 --websites shopping
python eval_results.py --experiment sgdr --model gpt-4.1 --metric autoeval
```

Use a different model for trajectory evaluation and SGDR induction:

```bash
python run_online.py \
  --experiment sgdr \
  --website shopping \
  --task_ids "21-25" \
  --model gpt-4o-mini \
  --eval_model gpt-4o
```

Run with a local vLLM backend:

```bash
# Terminal 1
bash serve_vllm.sh

# Terminal 2
source env.sh vllm
python run_online.py \
  --experiment sgdr \
  --website shopping \
  --task_ids "21-25"
```

In vLLM mode, `LLM_MODEL_NAME` from `env.sh` overrides CLI model arguments.

## Outputs

Typical SGDR outputs are:

```text
sgdr/results/sgdr_{model}/
  webarena.{id}/
    summary_info.json
    cleaned_steps.json
    {eval_model}_autoeval.json
  sgdr_logs/{id}.jsonl

sgdr/actions/_skill_lib/sgdr_{model}/
  {website}.jsonl
  _history/
```

`sgdr_logs/{id}.jsonl` records per-step retrieval information, including the goal, state summary, injected skills, and retrieval scores.

## Notes

- WebArena services should be reset between large model comparisons to avoid state carryover.
- Commercial backends may incur API costs for agent calls, induction, autoeval, and some WebArena built-in evaluators.
- Do not commit local host files, generated task configs, result directories, or API keys.

## License

This project is released under the license specified in [LICENSE](LICENSE).

## Citation

If you use this repository, please cite the paper linked above. BibTeX will be added when the final citation metadata is available.
