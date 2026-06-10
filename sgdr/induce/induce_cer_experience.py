from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re

from llm_client import llm_completion
from retrieval.cer_store import CERStore, DynamicsExperience, SkillExperience


_DYNAMICS_SYS = """You distill environment dynamics from a web agent trajectory.
Each dynamic should summarize one useful page/state:
- name
- description (what is on the page)
- usages (how it helps solve tasks)
- url

Keep them general and reusable.
If a page is already summarized in existing dynamics, skip it."""

_SKILLS_SYS = """You distill reusable decision-making skills from a web agent trajectory.
Break the overall goal into sub-goals; each sub-goal becomes one skill.

For every skill, output the name and 2-6 ordered steps. EACH step MUST contain
both a one-sentence natural-language description AND a concrete BrowserGym
action code block on the very next line, fenced with triple backticks.

Represent non-fixed elements (input strings, button strings, ids, names) with
descriptive placeholder variables (e.g. {forum_name}, {title}, {post_content},
{sort_criterion}, {submit_button_id}) — never hard-code task-specific values
or numeric bids that came from this single trajectory.

If a skill already exists in the provided existing skills list, skip it.

Example of one well-formed skill:
<skill>
Name: Sort products by sort criterion
Step 1: Click on the "Sort by" dropdown menu.
```click({sort_by_dropdown_id})```
Step 2: Select the desired sort criterion option from the dropdown.
```select_option({sort_by_dropdown_id}, {sort_criterion})```
</skill>"""


def _task_id_from_result_dir(result_dir: str) -> str:
    base = os.path.basename(result_dir)
    if "." not in base:
        return ""
    return base.split(".", 1)[1].split("_")[0]


def _load_goal(config_dir: str, task_id: str) -> str:
    if not task_id:
        return "Web navigation task."
    path = os.path.join(config_dir, f"{task_id}.json")
    if not os.path.exists(path):
        return "Web navigation task."
    cfg = json.load(open(path))
    return cfg.get("intent") or cfg.get("intent_template") or "Web navigation task."


def _step_files(result_dir: str) -> list[str]:
    names = [x for x in os.listdir(result_dir) if x.startswith("step_") and x.endswith(".pkl.gz")]
    names.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
    return [os.path.join(result_dir, x) for x in names]


def _safe_short_axtree(obs: dict, max_lines: int = 18) -> str:
    txt = (obs or {}).get("axtree_txt", "") or ""
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


def _serialize_trajectory(result_dir: str, max_steps: int = 30) -> str:
    rows = []
    for i, fp in enumerate(_step_files(result_dir)):
        if i >= max_steps:
            break
        try:
            step = pickle.load(gzip.open(fp, "rb"))
            obs = step.obs
        except Exception:
            continue
        if obs is None:
            continue
        idx = int(obs.get("active_page_index", 0) or 0)
        urls = obs.get("open_pages_urls", []) or []
        titles = obs.get("open_pages_titles", []) or []
        url = urls[idx] if 0 <= idx < len(urls) else ""
        title = titles[idx] if 0 <= idx < len(titles) else ""
        # Chosen action.
        action = getattr(step, "action", "") or ""
        err = obs.get("last_action_error", "")
        rows.append(
            f"[Step {i}] URL={url}\nTitle={title}\nAction={action}\n"
            f"Error={err}\nAXTree:\n{_safe_short_axtree(obs)}"
        )
    return "\n\n".join(rows)


def _extract_tag_items(text: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    out = []
    for m in pattern.findall(text or ""):
        item = m.strip()
        if item:
            out.append(item)
    return out


def _parse_dynamics(text: str, src_task_id: str) -> list[DynamicsExperience]:
    chunks = _extract_tag_items(text, "dynamic")
    out = []
    for ch in chunks:
        name = ""
        desc = ""
        usages = ""
        url = ""
        for line in ch.splitlines():
            s = line.strip()
            lower = s.lower()
            if lower.startswith("name:"):
                name = s.split(":", 1)[1].strip()
            elif lower.startswith("description:"):
                desc = s.split(":", 1)[1].strip()
            elif lower.startswith("usages:"):
                usages = s.split(":", 1)[1].strip()
            elif lower.startswith("url:"):
                url = s.split(":", 1)[1].strip()
        if name and desc:
            out.append(
                DynamicsExperience(
                    name=name,
                    description=desc,
                    usages=usages or "Navigate to this page for related operations.",
                    url=url,
                    src_task_id=src_task_id,
                )
            )
    return out


_STEP_HEADER_RE = re.compile(r"^\s*(?:[-*]\s*)?\*{0,2}step\s*\d+\*{0,2}\s*[:.\)]\s*", re.IGNORECASE)
_NAME_HEADER_RE = re.compile(r"^\s*(?:[-*]\s*)?\*{0,2}name\*{0,2}\s*[:.]\s*", re.IGNORECASE)


def _parse_skills(text: str, src_task_id: str) -> list[SkillExperience]:
    """Parse <skill> blocks."""
    chunks = _extract_tag_items(text, "skill")
    out = []
    for ch in chunks:
        lines = ch.splitlines()
        name = ""
        # Name line.
        body_start = 0
        for i, raw in enumerate(lines):
            if _NAME_HEADER_RE.match(raw):
                name = _NAME_HEADER_RE.sub("", raw).strip()
                body_start = i + 1
                break

        # Step blocks.
        steps: list[str] = []
        cur: list[str] = []
        cur_started = False
        for raw in lines[body_start:]:
            if _STEP_HEADER_RE.match(raw):
                if cur_started:
                    chunk = "\n".join(cur).strip()
                    if chunk:
                        steps.append(chunk)
                desc = _STEP_HEADER_RE.sub("", raw).strip()
                cur = [desc] if desc else []
                cur_started = True
            elif cur_started:
                cur.append(raw)
        if cur_started:
            chunk = "\n".join(cur).strip()
            if chunk:
                steps.append(chunk)

        if name and steps:
            out.append(SkillExperience(name=name, steps=steps, src_task_id=src_task_id))
    return out


def _format_existing_dynamics(items: list[DynamicsExperience], max_items: int = 120) -> str:
    rows = []
    for x in items[:max_items]:
        rows.append(f"- {x.name} | {x.url} | {x.description}")
    return "\n".join(rows) if rows else "(none)"


def _format_existing_skills(items: list[SkillExperience], max_items: int = 120) -> str:
    """Render compact existing skills."""
    rows = []
    for x in items[:max_items]:
        first_desc = x.steps[0].splitlines()[0].strip() if x.steps else ""
        rows.append(f"- {x.name} | {first_desc}")
    return "\n".join(rows) if rows else "(none)"


def distill_and_merge(
    model: str,
    result_dir: str,
    config_dir: str,
    store: CERStore,
) -> dict:
    task_id = _task_id_from_result_dir(result_dir)
    goal = _load_goal(config_dir, task_id)
    traj = _serialize_trajectory(result_dir)
    if not traj.strip():
        return {"dynamics": {"added": 0, "skipped": 0}, "skills": {"added": 0, "skipped": 0}, "task_id": task_id}

    dyn_user = (
        f"Task goal:\n{goal}\n\nTrajectory:\n{traj}\n\n"
        f"Existing dynamics:\n{_format_existing_dynamics(store.dynamics)}\n\n"
        "Output format:\n"
        "<dynamic>\n"
        "Name: ...\nDescription: ...\nUsages: ...\nURL: ...\n"
        "</dynamic>\n"
        "Repeat <dynamic> blocks for multiple items."
    )
    skill_user = (
        f"Task goal:\n{goal}\n\nTrajectory:\n{traj}\n\n"
        f"Existing skills:\n{_format_existing_skills(store.skills)}\n\n"
        "Output format (repeat <skill> blocks for multiple skills):\n"
        "<skill>\n"
        "Name: <skill name>\n"
        "Step 1: <one-sentence description>\n"
        "```<browsergym action call>```\n"
        "Step 2: <one-sentence description>\n"
        "```<browsergym action call>```\n"
        "</skill>\n"
        "Each Step MUST be immediately followed by a fenced action code block."
    )

    dyn_resp = llm_completion(
        model=model,
        messages=[
            {"role": "system", "content": _DYNAMICS_SYS},
            {"role": "user", "content": dyn_user},
        ],
        temperature=0.0,
    )
    sk_resp = llm_completion(
        model=model,
        messages=[
            {"role": "system", "content": _SKILLS_SYS},
            {"role": "user", "content": skill_user},
        ],
        temperature=0.0,
    )
    dyn_items = _parse_dynamics(dyn_resp.choices[0].message.content or "", task_id)
    sk_items = _parse_skills(sk_resp.choices[0].message.content or "", task_id)
    dyn_stats = store.add_dynamics(dyn_items)
    sk_stats = store.add_skills(sk_items)
    store.save()
    return {"dynamics": dyn_stats, "skills": sk_stats, "task_id": task_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default="config_files")
    parser.add_argument("--cer_store_path", type=str, required=True)
    args = parser.parse_args()

    store = CERStore(args.cer_store_path)
    stats = distill_and_merge(
        model=args.model,
        result_dir=args.result_dir,
        config_dir=args.config_dir,
        store=store,
    )
    print(f"[cer] distilled task={stats['task_id']} dynamics={stats['dynamics']} skills={stats['skills']}")


if __name__ == "__main__":
    main()
