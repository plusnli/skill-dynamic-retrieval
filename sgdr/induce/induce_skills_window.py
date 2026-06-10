"""Induce SGDR skills from cleaned trajectory windows."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
from typing import Iterable

from llm_client import llm_completion
from induce.utils import (
    extract_code_pieces,
    get_result_dirs,
    get_task_id,
)
from retrieval.skill_store import Skill, SkillStore


# Window enumeration.

def enumerate_windows(
    steps: list[str],
    lengths: Iterable[int] = (2, 3, 4, 5),
    stride: int = 1,
) -> list[dict]:
    """Return list of {"l": int, "t": int, "steps": [str, ...]}."""
    out = []
    H = len(steps)
    for l in lengths:
        if l > H:
            continue
        for t in range(0, H - l + 1, stride):
            out.append({"l": l, "t": t, "steps": steps[t : t + l]})
    return out


# LLM judge.

def _format_windows_for_prompt(windows: list[dict]) -> str:
    """Render windows as a numbered block for the LLM."""
    blocks = []
    for i, w in enumerate(windows):
        body = "\n".join(w["steps"])
        blocks.append(f"### Window {i} (l={w['l']}, start_t={w['t']})\n{body}")
    return "\n\n".join(blocks)


def _strip_md_fence(s: str) -> str:
    """Strip a markdown JSON fence."""
    s = s.strip()
    if s.startswith("```"):
        # Drop fence lines.
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _extract_json_array(text: str) -> list[dict]:
    """Best-effort parse of the model's JSON-array reply."""
    text = _strip_md_fence(text)
    # Direct parse.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Fallback span parse.
    m = re.search(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON array from model reply:\n{text[:500]}")


def judge_windows(
    windows: list[dict],
    sys_msg: str,
    instruction: str,
    task_text: str,
    model: str,
    batch_size: int = 16,
    temperature: float = 0.2,
) -> list[dict]:
    """Judge candidate windows."""
    results: list[dict] = [None] * len(windows)

    for batch_start in range(0, len(windows), batch_size):
        batch = windows[batch_start : batch_start + batch_size]
        rendered = _format_windows_for_prompt(batch)
        user_msg = (
            f"## Task instruction (for context only)\n{task_text}\n\n"
            f"## Action windows\n{rendered}"
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": instruction},
            {"role": "user", "content": user_msg},
        ]
        response = llm_completion(
            model=model, messages=messages, temperature=temperature, n=1,
        )
        reply = response.choices[0].message.content
        try:
            parsed = _extract_json_array(reply)
        except ValueError as e:
            print(f"[judge_windows] parse failed for batch starting at {batch_start}: {e}")
            for j in range(len(batch)):
                results[batch_start + j] = {"reusable": False}
            continue

        # Align by index.
        by_idx = {}
        for k, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            idx = item.get("window_idx", k)
            by_idx[idx] = item
        for j in range(len(batch)):
            results[batch_start + j] = by_idx.get(j, {"reusable": False})

    return results


# Code validation.

# Allowed primitive actions.
_ALLOWED_ACTIONS: frozenset[str] = frozenset({
    "click", "fill", "hover", "keyboard_press", "scroll",
    "tab_focus", "new_tab", "tab_close", "go_back", "go_forward",
    "goto", "send_msg_to_user", "report_infeasible",
    "report_infeasible_instructions", "select_option",
})

_MIN_BODY_CALLS = 2
_MAX_BODY_CALLS = 5


def validate_skill_code(code: str) -> tuple[bool, str | None, str | None]:
    """Validate generated skill code."""
    code = code.strip()
    if not code:
        return False, None, "empty"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, None, f"syntax: {e}"

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, None, f"expected exactly 1 top-level def, got {len(tree.body)} nodes"
    fdef: ast.FunctionDef = tree.body[0]

    n = len(fdef.body)
    if not (_MIN_BODY_CALLS <= n <= _MAX_BODY_CALLS):
        return False, fdef.name, f"body has {n} statements, want {_MIN_BODY_CALLS}-{_MAX_BODY_CALLS}"

    for stmt in fdef.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False, fdef.name, f"non-call statement in body: {ast.dump(stmt)[:80]}"
        call = stmt.value
        if not isinstance(call.func, ast.Name):
            # Reject method calls.
            return False, fdef.name, "method-call style not allowed; use bare action(...)"
        if call.func.id not in _ALLOWED_ACTIONS:
            return False, fdef.name, f"disallowed action '{call.func.id}'"

    return True, fdef.name, None


# Trajectory verification.

def _strip_python_prefix(code: str) -> str:
    code = code.strip()
    if code.startswith("python\n"):
        return code[len("python\n"):].strip()
    return code


def _extract_action_lines(step: str) -> list[str]:
    """Extract executable action lines from one cleaned trajectory step."""
    pieces = extract_code_pieces(step, start="```", end="```", do_split=False)
    if not pieces:
        pieces = [step]

    actions: list[str] = []
    for piece in pieces:
        piece = _strip_python_prefix(piece)
        for line in piece.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            actions.append(line)
    return actions


def _parse_call(src: str) -> ast.Call | None:
    """Parse one action line into an ast.Call, rejecting non-call snippets."""
    try:
        tree = ast.parse(src.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    call = tree.body[0].value
    return call if isinstance(call, ast.Call) else None


def _call_name(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else None


def validate_skill_call(skill_call: str, func_name: str) -> tuple[bool, str | None, str]:
    """Validate the generated skill call."""
    if not isinstance(skill_call, str) or not skill_call.strip():
        return False, None, "missing skill_call"

    lines = _extract_action_lines(skill_call)
    if len(lines) != 1:
        return False, None, f"skill_call must be one function call, got {len(lines)} lines"

    call_src = lines[0]
    call = _parse_call(call_src)
    if call is None:
        return False, None, f"skill_call is not a parseable function call: {call_src}"
    call_name = _call_name(call)
    if call_name != func_name:
        return False, None, f"skill_call invokes {call_name}, expected {func_name}"
    return True, call_src, "ok"


def _write_action_path(path: str, actions: list[str]) -> None:
    with open(path, "w") as f:
        for action in actions:
            f.write(f"```{action}```\n")


def _find_autoeval_json(result_dir: str) -> str | None:
    if not os.path.isdir(result_dir):
        return None
    for fname in os.listdir(result_dir):
        if fname.endswith("_autoeval.json"):
            return os.path.join(result_dir, fname)
    return None


def _parse_rewritten_trajectory(reply: str) -> list[str]:
    """Extract the rewritten trajectory."""
    lower = reply.lower()
    for marker in ("rewritten trajectories", "rewritten trajectory"):
        if marker in lower:
            reply = reply[lower.index(marker):]
            break

    blocks = extract_code_pieces(reply, start="```python", end="```", do_split=False)
    blocks = [b for b in blocks if "def " not in b]
    if not blocks:
        blocks = extract_code_pieces(reply, start="```", end="```", do_split=False)
        blocks = [b for b in blocks if "def " not in b]
    if not blocks:
        return []

    actions: list[str] = []
    for line in blocks[0].splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            actions.append(line)
    return actions


def _generate_test_trajectory(
    skill: Skill,
    steps: list[str],
    task_template: str,
    task_intent: str,
    model: str,
    temperature: float,
    sys_msg: str,
    instruction: str,
) -> tuple[list[str] | None, str]:
    """Generate a rewritten trajectory."""
    original = "\n".join(steps)
    user_msg = (
        f"## Task Template\n{task_template}\n\n"
        f"## Instantiated Task Instruction\n{task_intent}\n\n"
        f"### Example {skill.src_task_id}: {task_intent}\n"
        f"## Original Trajectory\n{original}\n\n"
        f"## Reusable Function\n```python\n{skill.code.strip()}\n```"
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": instruction},
        {"role": "user", "content": user_msg},
    ]
    try:
        response = llm_completion(
            model=model, messages=messages, temperature=temperature, n=1,
        )
    except Exception as e:  # noqa: BLE001 — induction is best-effort
        return None, f"test-trajectory generation failed: {e}"
    reply = response.choices[0].message.content or ""

    action_lines = _parse_rewritten_trajectory(reply)
    if not action_lines:
        return None, "no '## Rewritten Trajectory' block produced"
    if not any(skill.func_name in line for line in action_lines):
        return None, "rewritten trajectory does not call the induced function"
    return action_lines, "ok"


def verify_skill_via_test(
    skill: Skill,
    steps: list[str],
    result_dir: str,
    task_template: str,
    task_intent: str,
    website: str,
    results_dir: str,
    gen_model: str,
    eval_model: str,
    temperature: float,
    sys_msg: str,
    instruction: str,
    verify_dir: str,
    solve_timeout: int,
    eval_timeout: int,
    keep_results: bool = False,
) -> tuple[bool, str]:
    """Replay-check a generated skill."""
    cid = get_task_id(result_dir)

    rewritten_actions, reason = _generate_test_trajectory(
        skill, steps, task_template, task_intent, gen_model, temperature,
        sys_msg, instruction,
    )
    if rewritten_actions is None:
        return False, reason

    os.makedirs(verify_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", skill.func_name)
    # Temp path label.
    t = skill.window.get("t", 0)
    l = skill.window.get("l", 0)
    verify_stem = f"{cid}_{os.getpid()}_{safe_name}_{t}_{l}"
    action_path = os.path.join(verify_dir, f"{verify_stem}.txt")
    verify_result_name = f"webarena.{cid}_sgdr_verify_{os.getpid()}_{safe_name}_{t}_{l}"
    verify_result_dir = os.path.join(results_dir, verify_result_name)

    live_action_path = os.path.join("actions", f"{website}.py")
    try:
        original_action_file = open(live_action_path, "r").read()
    except OSError as e:
        return False, f"could not read {live_action_path}: {e}"

    _write_action_path(action_path, rewritten_actions)
    try:
        with open(live_action_path, "w") as f:
            f.write(original_action_file.rstrip() + "\n\n\n" + skill.code.rstrip() + "\n")

        # Cover full replay length.
        max_steps = max(10, len(rewritten_actions) + 1)
        solve_cmd = [
            "python", "run_demo.py",
            "--websites", website,
            "--headless",
            "--task_name", f"webarena.{cid}",
            "--action_path", action_path,
            "--results_dir", results_dir,
            "--rename_to", verify_result_name,
            "--max_steps", str(max_steps),
        ]
        solve = subprocess.run(
            solve_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=solve_timeout,
        )
        if solve.returncode != 0:
            return False, f"replay failed exit={solve.returncode}: {solve.stdout[-800:]}"

        eval_cmd = [
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", verify_result_dir,
            "--model", eval_model,
        ]
        ev = subprocess.run(
            eval_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=eval_timeout,
        )
        if ev.returncode != 0:
            return False, f"autoeval failed exit={ev.returncode}: {ev.stdout[-800:]}"

        eval_path = _find_autoeval_json(verify_result_dir)
        if not eval_path:
            return False, "autoeval output missing"
        try:
            is_success = json.load(open(eval_path))[0].get("rm") is True
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            return False, f"autoeval output parse failed: {e}"
        if not is_success:
            return False, "rewritten trajectory judged unsuccessful"

        validity = subprocess.run(
            [
                "python", "utils/calc_valid_steps.py",
                "--result_dir", verify_result_dir,
                "--action_names", skill.func_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=eval_timeout,
        )
        if validity.returncode != 0:
            return False, f"validity check failed exit={validity.returncode}: {validity.stdout[-800:]}"
        # Check skill usage.
        if validity.stdout.strip().splitlines()[-1].strip() == "False":
            return False, f"skill call did not pass validity check: {validity.stdout[-800:]}"

        return True, "verified"
    except subprocess.TimeoutExpired as e:
        return False, f"verification timed out after {e.timeout}s"
    finally:
        with open(live_action_path, "w") as f:
            f.write(original_action_file)
        if not keep_results:
            if os.path.exists(action_path):
                os.remove(action_path)
            if os.path.isdir(verify_result_dir):
                shutil.rmtree(verify_result_dir, ignore_errors=True)


def verify_skills_for_trajectory(
    skills: list[Skill],
    result_dir: str,
    args,
) -> list[Skill]:
    if not skills:
        return []
    cleaned_path = os.path.join(result_dir, "cleaned_steps.json")
    steps: list[str] = json.load(open(cleaned_path))

    # Task context.
    cid = get_task_id(result_dir)
    config = json.load(open(os.path.join(args.config_dir, f"{cid}.json")))
    task_template = config.get("intent_template") or config.get("intent", "")
    task_intent = config.get("intent") or task_template
    sys_msg = open(args.verify_sys_msg_path).read()
    instruction = open(args.verify_rewrite_path).read()

    verified: list[Skill] = []
    for skill in skills:
        ok, reason = verify_skill_via_test(
            skill=skill,
            steps=steps,
            result_dir=result_dir,
            task_template=task_template,
            task_intent=task_intent,
            website=args.website,
            results_dir=args.results_dir,
            gen_model=args.model,
            eval_model=args.eval_model or args.model,
            temperature=args.temperature,
            sys_msg=sys_msg,
            instruction=instruction,
            verify_dir=args.verify_dir,
            solve_timeout=args.verify_timeout,
            eval_timeout=args.verify_eval_timeout,
            keep_results=args.keep_verify_results,
        )
        status = "verified" if ok else "rejected"
        print(
            f"  ↳ verify {status}: {skill.func_name} "
            f"(l={skill.window.get('l')}, t={skill.window.get('t')}): {reason}"
        )
        if ok:
            verified.append(skill)
    print(f"[induce_skills_window] verified {len(verified)}/{len(skills)} skills")
    return verified


# Pipeline.

def induce_one_trajectory(
    result_dir: str,
    config_dir: str,
    sys_msg: str,
    instruction: str,
    model: str,
    lengths: tuple[int, ...],
    batch_size: int,
    website: str,
) -> list[Skill]:
    """Extract candidate skills from one cleaned trajectory."""
    cid = get_task_id(result_dir)
    config_path = os.path.join(config_dir, f"{cid}.json")
    config = json.load(open(config_path))
    task_text = config.get("intent_template") or config.get("intent", "")

    cleaned_path = os.path.join(result_dir, "cleaned_steps.json")
    if not os.path.exists(cleaned_path):
        print(f"[induce_skills_window] {cleaned_path} missing, skip")
        return []
    steps: list[str] = json.load(open(cleaned_path))
    if len(steps) < min(lengths):
        return []

    windows = enumerate_windows(steps, lengths=lengths)
    print(f"[induce_skills_window] task {cid}: {len(steps)} steps → {len(windows)} windows")
    if not windows:
        return []

    judged = judge_windows(
        windows, sys_msg, instruction, task_text,
        model=model, batch_size=batch_size,
    )

    skills: list[Skill] = []
    for w, j in zip(windows, judged):
        if not isinstance(j, dict) or not j.get("reusable"):
            continue
        code = j.get("code", "")
        desc = j.get("description", "").strip()
        if not desc or not code:
            continue
        ok, func_name, reason = validate_skill_code(code)
        if not ok:
            print(f"  ↳ window(l={w['l']},t={w['t']}) rejected: {reason}")
            continue
        # Check LLM name.
        if j.get("func_name") and j["func_name"] != func_name:
            print(f"  ↳ func_name mismatch (LLM said {j['func_name']}, parsed {func_name})")
        call_ok, skill_call, call_reason = validate_skill_call(
            j.get("skill_call", ""),
            func_name,
        )
        if not call_ok:
            print(f"  ↳ window(l={w['l']},t={w['t']}) skill_call ignored: {call_reason}")
            skill_call = ""
        skills.append(Skill(
            description=desc,
            code=code,
            func_name=func_name,
            website=website,
            src_task_id=cid,
            window={"l": w["l"], "t": w["t"], "skill_call": skill_call},
        ))
    print(f"[induce_skills_window] task {cid}: kept {len(skills)} skills")
    return skills


# CLI.

def _default_skill_path(website: str, tag: str) -> str:
    return os.path.join("actions", "_skill_lib", tag, f"{website}.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for the ASI-style rewrite step "
                             "that generates each skill's test trajectory. "
                             "Matches ASI induce_actions' default (1.0).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 3, 4, 5])

    parser.add_argument("--sys_msg_path", type=str,
                        default="induce/prompt/system_message_window.txt")
    parser.add_argument("--instruction_path", type=str,
                        default="induce/prompt/window_judge.txt")

    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "admin", "reddit", "gitlab", "map"])
    parser.add_argument("--config_dir", type=str, default="config_files")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--result_id_list", type=str, nargs="+", default=None)
    parser.add_argument("--template_id", type=str, default=None)

    parser.add_argument("--skill_store_path", type=str, default=None,
                        help="Path to skills JSONL.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag for default skill_store_path.")

    parser.add_argument("--dry_run", action="store_true",
                        help="Print extracted skills + write decisions, do not persist.")

    parser.add_argument("--tau_high", type=float, default=None,
                        help="Skip a candidate if max desc cosine >= tau_high. "
                             "Only used with --dedup_embed.")
    parser.add_argument("--tau_low", type=float, default=None,
                        help="Tag near_dup_of if tau_low <= max cosine < tau_high. "
                             "Only used with --dedup_embed.")
    parser.add_argument("--dedup_embed", action="store_true",
                        help="Enable legacy description-similarity dedup at write "
                             "time. Off by default for SGDR reproduction; only "
                             "func_name collision applies when unset.")
    parser.add_argument("--verify_skills", action="store_true",
                        help="Verify each induced skill the ASI way: the LLM "
                             "rewrites the original trajectory using the induced "
                             "function, that trajectory is replayed, and only "
                             "autoeval-successful skills are kept. SGDR enables "
                             "this from run_online.py.")
    parser.add_argument("--eval_model", type=str, default=None,
                        help="Model for verification autoeval. Defaults to --model.")
    parser.add_argument("--verify_sys_msg_path", type=str,
                        default="induce/prompt/system_message.txt",
                        help="System prompt for the ASI-style rewrite step that "
                             "generates each skill's test trajectory.")
    parser.add_argument("--verify_rewrite_path", type=str,
                        default="induce/prompt/window_verify_rewrite.txt",
                        help="Instruction prompt asking the LLM to rewrite the "
                             "original trajectory using the induced function.")
    parser.add_argument("--verify_dir", type=str,
                        default=os.path.join("debug_actions", "sgdr_verify"),
                        help="Directory for temporary verification action paths.")
    parser.add_argument("--verify_timeout", type=int, default=300,
                        help="Timeout in seconds for each substituted-trajectory replay.")
    parser.add_argument("--verify_eval_timeout", type=int, default=180,
                        help="Timeout in seconds for each verification autoeval/check.")
    parser.add_argument("--keep_verify_results", action="store_true",
                        help="Keep temporary *_sgdr_verify_* result dirs and action "
                             "files for debugging.")

    args = parser.parse_args()

    if args.skill_store_path is None:
        if args.tag is None:
            from run_online import skill_tag, sanitize_model_name
            args.tag = skill_tag("sgdr", args.model)
        args.skill_store_path = _default_skill_path(args.website, args.tag)

    sys_msg = open(args.sys_msg_path).read()
    # Judge prompt.
    instruction = open(args.instruction_path).read()

    result_dirs = get_result_dirs(
        args.results_dir, args.result_id_list, args.template_id, args.config_dir,
    )
    print(f"[induce_skills_window] {len(result_dirs)} trajectories → {args.skill_store_path}")

    # Store config.
    from retrieval.embedder import Embedder

    embedder = Embedder() if args.dedup_embed else None
    legacy_tau_low_default = 0.65
    store_kwargs = {
        "tau_high": args.tau_high if args.dedup_embed else None,
        "tau_low": (
            args.tau_low
            if args.dedup_embed and args.tau_low is not None
            else legacy_tau_low_default
            if args.dedup_embed
            else None
        ),
    }
    if args.dry_run:
        # Load only.
        store = SkillStore(args.skill_store_path, embedder=embedder, **store_kwargs)
    else:
        store = SkillStore(args.skill_store_path, embedder=embedder, **store_kwargs)
    semantic_dedup = (
        embedder is not None
        and (store.tau_high is not None or store.tau_low is not None)
    )
    print(f"[induce_skills_window] store loaded: {len(store)} existing skills "
          f"(tau_high={store.tau_high}, tau_low={store.tau_low}, "
          f"semantic_dedup={'on' if semantic_dedup else 'off'})")

    agg = {"added": 0, "skipped_name": 0, "skipped_dup": 0,
           "skipped_same_task": 0, "near_dup_kept": 0,
           "rejected_verify": 0}
    for rd in result_dirs:
        skills = induce_one_trajectory(
            rd, args.config_dir, sys_msg, instruction,
            model=args.model, lengths=tuple(args.lengths),
            batch_size=args.batch_size, website=args.website,
        )
        if args.verify_skills:
            n_before_verify = len(skills)
            skills = verify_skills_for_trajectory(skills, rd, args)
            agg["rejected_verify"] += n_before_verify - len(skills)
        if args.dry_run:
            print(f"\n[dry_run] candidates from {rd}:")
            for s in skills:
                print(f"--- {s.func_name} ({s.src_task_id}, l={s.window['l']}, t={s.window['t']}) ---")
                print(f"description: {s.description}")
                print(f"skill_call: {s.window.get('skill_call', '')}")
                print(s.code)
                print()
        stats = store.add_many(skills)
        for k_, v_ in stats.items():
            agg[k_] = agg.get(k_, 0) + v_
        print(f"  ↳ {rd}: {stats}")

    if not args.dry_run:
        store.save()
        if embedder is not None:
            embedder.flush()
    print(f"\n[induce_skills_window] aggregate: {agg}")
    print(f"[induce_skills_window] final store size: {len(store)} "
          f"({'NOT WRITTEN' if args.dry_run else args.skill_store_path})")


if __name__ == "__main__":
    main()
