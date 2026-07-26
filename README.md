# Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval

[arXiv](https://arxiv.org/pdf/2606.04391)
[License: CC BY-SA 4.0](LICENSE)
[Python 3.10](https://www.python.org/)

This is the implementation of paper **[Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval](https://arxiv.org/pdf/2606.04391)**.

This repository uses [WebArena](https://webarena.dev/og/) as the web agent environment. You will need to first install and configure the docker environments from [https://github.com/web-arena-x/webarena/tree/main/environment_docker](https://github.com/web-arena-x/webarena/tree/main/environment_docker).

## Repository Organization

```text
skill-dynamic-retrieval/
  browsergym/          BrowserGym/WebArena dependencies used by the project
  sgdr/                Main Program: State-Grounded Dynamic Retrieval (SGDR)
    actions/           Base action sets and learned skill libraries
    autoeval/          Trajectory evaluation utilities
    config_files/      WebArena task configuration template and generator
    induce/            Skill induction pipelines
    retrieval/         Skill retrieval
    workflows/         Workflow-memory files for baseline AWM
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

`host.local.sh` is ignored by Git because it is machine-specific. 

Generated task configs under `config_files/*.json` are also ignored because they embed local service URLs.

Set your OpenAI API key in the shell or in a local ignored `.env` file:

```bash
export OPENAI_API_KEY="your-api-key"
```

Then load the runtime environment and generate WebArena task configs (dataset):

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
sgdr        State-Grounded Dynamic Retrieval
awm         Baseline: Agent Workflow Memory (AWM)
asi         Baseline: Agent Skill Induction (ASI)
cer_online  Baseline: Contextual Experience Replay (CER); Online version
```

Allowed website domains are `shopping`, `admin`, `reddit`, `gitlab`, and `map`.

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

This project is released under the **Creative Commons Attribution-ShareAlike 4.0
International License (CC BY-SA 4.0)** — see [LICENSE](LICENSE). If you distribute a
modified version, the ShareAlike condition requires you to release it under the same
license and to credit the original authors.

The repository contains code derived from two upstream projects, each with its own terms:

| Path | Upstream | License |
|---|---|---|
| `sgdr/` | [ASI — Agent Skill Induction](https://github.com/zorazrw/agent-skill-induction) | CC BY-SA 4.0 |
| `browsergym/` | [BrowserGym](https://github.com/ServiceNow/BrowserGym) v0.10.2, © ServiceNow | Apache-2.0 |

`sgdr/` is a derivative work of ASI; because ASI is licensed under CC BY-SA 4.0, the
ShareAlike condition requires this repository to carry the same license.

`browsergym/` is a vendored **and modified** copy of BrowserGym. It stays under the
Apache License 2.0 and is *not* relicensed — see [browsergym/LICENSE](browsergym/LICENSE)
for the license text and [browsergym/NOTICE](browsergym/NOTICE) for the list of changes
made to the original sources.

## Citation

If you find this paper / repository helpful to your research, please consider citing it as follows:

```bibtex
@article{li2026online,
  title={Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval},
  author={Li, Jiaxi and Deng, Ke and Wang, Yun and Huang, Jingyuan and Shi, Yucheng and Tan, Qiaoyu and Lu, Jin and Liu, Ninghao},
  journal={arXiv preprint arXiv:2606.04391},
  year={2026}
}
```