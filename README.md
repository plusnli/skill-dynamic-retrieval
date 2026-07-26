# Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval

[arXiv](https://arxiv.org/pdf/2606.04391)
[License: CC BY-SA 4.0](LICENSE)
[Python 3.10](https://www.python.org/)

This is the implementation of paper **[Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval](https://arxiv.org/pdf/2606.04391)**.

This repository uses [WebArena](https://webarena.dev/og/) as the web agent environment. You will need to first install and configure the docker environments from [https://github.com/web-arena-x/webarena/tree/main/environment_docker](https://github.com/web-arena-x/webarena/tree/main/environment_docker).

## 📁 Repository Organization

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

## ⚙️ Setup

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



## 🚀 Quick Start

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

Allowed website domains are `shopping`, `admin`, `reddit`, `gitlab`, and `map`. We remove `Wiki` as it involves activity across multiple websites.

## 🔄 SGDR Pipeline

For each task, `run_online.py --experiment sgdr` performs:

1. **Solve**: run the web agent with dynamic skill retrieval.
2. **Evaluate**: llm-as-a-judge whether the trajectory completed the task.
3. **Induce**: synthesize reusable skills from successful cleaned trajectories with sliding windows and verification.
4. **Update**: append new skills to the skill library, leaving them available for future tasks.

The skill library is stored under:

```text
sgdr/actions/_skill_lib/sgdr_{model}/{website}.jsonl
```

At the start of a new SGDR run, an existing library for the same model and website will be archived under `_history/`.

## 🛠️ Running Commands



### Run all tasks of one website

`env.sh` defines the full task-ID list for each website as `TASK_IDS_SHOPPING`, `TASK_IDS_ADMIN`, `TASK_IDS_REDDIT`, `TASK_IDS_GITLAB`, and `TASK_IDS_MAP`, so a full-website run just passes the matching variable to `--task_ids`. 

**With a commercial backend (default, LiteLLM → OpenAI):**

```bash
cd skill-dynamic-retrieval/sgdr
source env.sh

python run_online.py \
  --experiment sgdr \
  --website shopping \
  --task_ids "$TASK_IDS_SHOPPING" \
  --model gpt-4.1
```

**With a local vLLM backend (open-source model):**

```bash
# Terminal 1 — start the server. The model comes from VLLM_MODEL_NAME in env.sh
# (default: Qwen/Qwen3-4B).
bash serve_vllm.sh

# Terminal 2 — run the experiment
source env.sh vllm

python run_online.py \
  --experiment sgdr \
  --website shopping \
  --task_ids "$TASK_IDS_SHOPPING"
```

In vLLM mode, `LLM_MODEL_NAME` from `env.sh` overrides the CLI `--model` and `--eval_model` arguments, so the served model is always the one that is used.

### Evaluate with environment reward

WebArena ships its own programmatic evaluator for every task, and its verdict (`cum_reward > 0`) is the objective accuracy reported in the paper. This is the default `--metric env`:

```bash
python eval_results.py \
  --experiment sgdr \
  --model gpt-4.1 \
  --websites shopping \
  --metric env
```

Omit `--websites` to print a per-website breakdown plus the overall total across all five domains. Under a vLLM backend, pass the served model to `--model` (for example `--model Qwen/Qwen3-4B`), since the results directory is named after it.

The other metric option `--metric autoeval` is used for diagnosis rather than for reporting: it shows the LLM-as-a-judge verdict that drives skill induction during the program run. 

## 📊 Outputs

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

## 📝 Notes

- WebArena services should be reset between experiment runs on the same website.
- Commercial backends may be set for API calls for some WebArena and BrowserGym built-in functions.



## 📄 License

This project is released under CC BY-SA 4.0 — see [LICENSE](LICENSE).

The `browsergym/` directory is a vendored and modified copy of
[BrowserGym](https://github.com/ServiceNow/BrowserGym) and remains under Apache-2.0;
see [browsergym/LICENSE](browsergym/LICENSE) and [browsergym/NOTICE](browsergym/NOTICE).

## 📌 Citation

If you find this paper / repository helpful to your research, please consider citing it as follows:

```bibtex
@article{li2026online,
  title={Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval},
  author={Li, Jiaxi and Deng, Ke and Wang, Yun and Huang, Jingyuan and Shi, Yucheng and Tan, Qiaoyu and Lu, Jin and Liu, Ninghao},
  journal={arXiv preprint arXiv:2606.04391},
  year={2026}
}
```

