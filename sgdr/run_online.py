import os
import json
import shutil
import time
import filecmp
import argparse
import subprocess
from subprocess import Popen
from tqdm import tqdm

def parse_task_ids(task_id_str: str) -> list[str]:
    chunks = [c.strip() for c in task_id_str.split(",")]
    task_id_list = []
    for c in chunks:
        parts = [n.strip() for n in c.split("-")]
        if len(parts) == 1:
            task_id_list.append(parts[0])
        else:
            s, e = int(parts[0]), int(parts[1])
            task_id_list.extend([str(i) for i in range(s, e+1)])
    return task_id_list

def sanitize_model_name(name: str) -> str:
    """Make a model name path-safe."""
    return name.replace("/", "_")

_SKILL_SPECS = [("actions", "{w}.py"), ("workflows", "{w}.txt")]

# SGDR JSONL library.
_SGDR_SKILL_BASE = "actions/_skill_lib"
_CER_BASE = "actions/_cer_lib"

def skill_tag(experiment: str, model: str) -> str:
    return f"{experiment}_{sanitize_model_name(model)}"


def induce_isolation_args(experiment: str, model: str, website: str) -> list[str]:
    """Keep induction caches separate."""
    tag = skill_tag(experiment, model)
    return [
        "--outputs_root", os.path.join("outputs", tag),
        "--write_tests_dir", os.path.join("debug_actions", tag, website),
    ]


def sgdr_skill_path(website: str, tag: str) -> str:
    """SGDR skill library path."""
    return os.path.join(_SGDR_SKILL_BASE, tag, f"{website}.jsonl")


def cer_buffer_path(website: str, tag: str) -> str:
    """CER replay buffer path."""
    return os.path.join(_CER_BASE, tag, f"{website}.json")


def _archive_to_history(tag_path: str, baseline: str):
    """Archive a non-baseline file."""
    if not os.path.exists(tag_path):
        return
    if filecmp.cmp(tag_path, baseline, shallow=False):
        os.remove(tag_path)
        return
    tag_dir = os.path.dirname(tag_path)
    fname   = os.path.basename(tag_path)
    history = os.path.join(tag_dir, "_history")
    os.makedirs(history, exist_ok=True)
    stem, ext = os.path.splitext(fname)
    ts  = time.strftime("%Y%m%dT%H%M%S")
    dst = os.path.join(history, f"{stem}_{ts}{ext}")
    if os.path.exists(dst):
        dst = os.path.join(history, f"{stem}_{ts}_{int(time.time()*1000) % 1000:03d}{ext}")
    shutil.move(tag_path, dst)
    print(f"[skills] archived {tag_path} → {dst}")


def setup_skill_files(website: str, tag: str):
    """Prepare per-run action and workflow files."""
    for base_dir, fname_tmpl in _SKILL_SPECS:
        fname     = fname_tmpl.format(w=website)
        baseline  = os.path.join(base_dir, "_baseline", fname)
        tag_dir   = os.path.join(base_dir, tag)
        tag_path  = os.path.join(tag_dir, fname)
        live_path = os.path.join(base_dir, fname)
        os.makedirs(tag_dir, exist_ok=True)

        if not os.path.exists(baseline):
            raise FileNotFoundError(
                f"Baseline missing: {baseline}. Create it once before running."
            )

        _archive_to_history(tag_path, baseline)
        shutil.copy2(baseline, tag_path)
        print(f"[skills] seeded {tag_path} ← {baseline}")

        # Replace symlinks too.
        if os.path.lexists(live_path):
            os.remove(live_path)
        os.symlink(os.path.join(tag, fname), live_path)
        print(f"[skills] {live_path} → {tag}/{fname}")


def setup_sgdr_files(
    website: str,
    tag: str,
    reuse: bool = False,
    label: str = "sgdr",
) -> str:
    """Prepare the SGDR skill library."""
    base = os.path.join(_SGDR_SKILL_BASE, tag)
    fname = f"{website}.jsonl"
    path = os.path.join(base, fname)
    os.makedirs(base, exist_ok=True)

    if reuse:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(
                f"--reuse_skill_lib set but {path} is missing or empty. "
                f"Run {label} normally at least once before reusing."
            )
        print(f"[{label}] reusing existing library: {path} "
              f"({sum(1 for _ in open(path))} skills, frozen)")
    else:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            history = os.path.join(base, "_history")
            os.makedirs(history, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%S")
            dst = os.path.join(history, f"{website}_{ts}.jsonl")
            if os.path.exists(dst):
                dst = os.path.join(history, f"{website}_{ts}_{int(time.time()*1000) % 1000:03d}.jsonl")
            shutil.move(path, dst)
            print(f"[{label}] archived {path} → {dst}")

        open(path, "w").close()
        print(f"[{label}] seeded blank: {path}")

    # Reset base actions.
    live_actions = os.path.join("actions", f"{website}.py")
    baseline_actions = os.path.join("actions", "_baseline", f"{website}.py")
    if not os.path.exists(baseline_actions):
        raise FileNotFoundError(
            f"Baseline missing: {baseline_actions}. Create it once before running."
        )
    if os.path.lexists(live_actions):
        os.remove(live_actions)  # also breaks dangling symlinks
    shutil.copy2(baseline_actions, live_actions)
    print(f"[{label}] reset {live_actions} ← {baseline_actions}")

    return path


def setup_cer_files(website: str, tag: str, reuse: bool = False) -> str:
    """Prepare the CER replay buffer."""
    base = os.path.join(_CER_BASE, tag)
    fname = f"{website}.json"
    path = os.path.join(base, fname)
    os.makedirs(base, exist_ok=True)

    if reuse and os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[cer] reusing existing replay buffer: {path}")
    else:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            history = os.path.join(base, "_history")
            os.makedirs(history, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%S")
            dst = os.path.join(history, f"{website}_{ts}.json")
            if os.path.exists(dst):
                dst = os.path.join(history, f"{website}_{ts}_{int(time.time()*1000) % 1000:03d}.json")
            shutil.move(path, dst)
            print(f"[cer] archived {path} → {dst}")

        with open(path, "w") as f:
            json.dump({"dynamics": [], "skills": []}, f)
        print(f"[cer] seeded blank: {path}")

    live_actions = os.path.join("actions", f"{website}.py")
    baseline_actions = os.path.join("actions", "_baseline", f"{website}.py")
    if not os.path.exists(baseline_actions):
        raise FileNotFoundError(
            f"Baseline missing: {baseline_actions}. Create it once before running."
        )
    if os.path.lexists(live_actions):
        os.remove(live_actions)
    shutil.copy2(baseline_actions, live_actions)
    print(f"[cer] reset {live_actions} ← {baseline_actions}")

    return path


def autoeval_path(result_dir: str) -> str:
    """Return the autoeval JSON path."""
    return os.path.join(result_dir, f"{sanitize_model_name(args.eval_model)}_autoeval.json")

def load_json_safe(path: str, tid: str, label: str):
    """Load JSON or skip the task."""
    if not os.path.exists(path):
        print(f"[webarena.{tid}] {label}: expected file not found: {path}, skipping task")
        return None
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[webarena.{tid}] {label}: failed to parse {path}: {e}, skipping task")
        return None

def run_step(cmd: list[str], label: str, tid: str, timeout: int = None,
             env: dict | None = None) -> bool:
    """Run one subprocess step."""
    process = Popen(cmd, env=env)
    try:
        process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        print(f"[webarena.{tid}] {label}: timed out after {timeout}s, skipping task")
        return False
    if process.returncode != 0:
        print(f"[webarena.{tid}] {label}: failed (exit code {process.returncode}), skipping task")
        return False
    return True


def is_task_completed(tid: str, experiment: str) -> bool:
    """Check resume status."""
    result_dir = f"{args.results_dir}/webarena.{tid}"
    summary_path = os.path.join(result_dir, "summary_info.json")
    if not os.path.exists(summary_path):
        return False
    try:
        n_steps = json.load(open(summary_path)).get("n_steps")
    except (json.JSONDecodeError, OSError):
        return False
    if isinstance(n_steps, int) and n_steps < 3:
        return True
    return os.path.exists(autoeval_path(result_dir))


# ── Run tracker ──────────────────────────────────────────────

class RunTracker:
    """Track task outcomes."""

    def __init__(self):
        self.succeeded = []        # completed all steps
        self.failed = []           # errored at some step
        self.skipped = []          # already had results (resume)
        self.filtered_steps = []   # filtered: n_steps < 3
        self.filtered_eval = []    # filtered: eval incorrect

    @property
    def filtered(self):
        return self.filtered_steps + self.filtered_eval

    def mark(self, tid: str, status: str):
        getattr(self, status).append(tid)

    def summary(self):
        total = len(self.succeeded) + len(self.failed) + len(self.skipped) + len(self.filtered)
        print("\n" + "=" * 60)
        print(f"Run summary: {total} tasks")
        print(f"  Succeeded : {len(self.succeeded)}")
        print(f"  Failed    : {len(self.failed)}")
        print(f"  Skipped   : {len(self.skipped)} (already completed)")
        print(f"  Filtered  : {len(self.filtered)} (n_steps < 3 or eval incorrect)")
        print(f"    n_steps < 3    : {len(self.filtered_steps)}")
        print(f"    eval incorrect : {len(self.filtered_eval)}")
        if self.failed:
            print(f"  Failed IDs: {', '.join(self.failed)}")
        print("=" * 60)

tracker = RunTracker()

# Solve timeout.
SOLVE_TIMEOUT = 300

# %% AWM

def run_awm():
    task_id_list = parse_task_ids(args.task_ids)

    for tid in tqdm(task_id_list, desc="awm", unit="task"):
        if args.resume and is_task_completed(tid, "awm"):
            print(f"[webarena.{tid}] already completed, skipping")
            tracker.mark(tid, "skipped")
            continue

        # Solve.
        if not run_step([
            "python", "run_demo.py",
            "--task_name", f"webarena.{tid}",
            "--model_name", args.model,
            "--results_dir", args.results_dir,
            "--memory_path", f"workflows/{args.website}.txt",
            "--headless",
        ], "solve", tid, timeout=SOLVE_TIMEOUT):
            tracker.mark(tid, "failed")
            continue

        # Evaluate.
        if not run_step([
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--model", args.eval_model,
        ], "autoeval", tid):
            tracker.mark(tid, "failed")
            continue
        path = autoeval_path(f"{args.results_dir}/webarena.{tid}")
        data = load_json_safe(path, tid, "autoeval")
        if data is None:
            tracker.mark(tid, "failed")
            continue
        is_correct = data[0]["rm"]
        if not is_correct:
            tracker.mark(tid, "filtered_eval")
            continue

        # Induce workflows.
        if not run_step([
            "python", "utils/calc_valid_steps.py",
            "--clean_and_store", "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--model", args.model,
        ], "clean", tid):
            tracker.mark(tid, "failed")
            continue

        if not run_step([
            "python", "-m", "induce.induce_memory",
            "--website", args.website,
            "--results_dir", args.results_dir,
            "--result_id_list", tid,
            "--model", args.model,
            *induce_isolation_args(args.experiment, args.model, args.website),
        ], "induce_memory", tid):
            tracker.mark(tid, "failed")
            continue

        tracker.mark(tid, "succeeded")

# %% ASI
def run_asi():
    """ASI loop (solve -> autoeval -> clean -> induce)."""
    task_id_list = parse_task_ids(args.task_ids)
    for tid in tqdm(task_id_list, desc=args.experiment, unit="task"):
        if args.resume and is_task_completed(tid, args.experiment):
            print(f"[webarena.{tid}] already completed, skipping")
            tracker.mark(tid, "skipped")
            continue

        # Solve.
        if not run_step([
            "python", "run_demo.py",
            "--task_name", f"webarena.{tid}",
            "--model_name", args.model,
            "--results_dir", args.results_dir,
            "--websites", args.website,
            "--headless"
        ], "solve", tid, timeout=SOLVE_TIMEOUT):
            tracker.mark(tid, "failed")
            continue
        path = f"{args.results_dir}/webarena.{tid}/summary_info.json"
        data = load_json_safe(path, tid, "solve")
        if data is None:
            tracker.mark(tid, "failed")
            continue
        if data["n_steps"] < 3:
            tracker.mark(tid, "filtered_steps")
            continue

        # Evaluate.
        if not run_step([
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--model", args.eval_model,
            "--eval_source", "state",
        ], "autoeval", tid):
            tracker.mark(tid, "failed")
            continue
        path = autoeval_path(f"{args.results_dir}/webarena.{tid}")
        data = load_json_safe(path, tid, "autoeval")
        if data is None:
            tracker.mark(tid, "failed")
            continue
        if not data[0]["rm"]:
            tracker.mark(tid, "filtered_eval")
            continue

        # Clean.
        if not run_step([
            "python", "utils/calc_valid_steps.py",
            "--clean_and_store", "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--model", args.model,
        ], "clean", tid):
            tracker.mark(tid, "failed")
            continue

        # Induce actions.
        run_step([
            "python", "-m", "induce.induce_actions",
            "--website", args.website,
            "--results_dir", args.results_dir,
            "--result_id_list", tid,
            "--model", args.model,
            *induce_isolation_args(args.experiment, args.model, args.website),
        ], "induce_actions", tid, timeout=200)

        tracker.mark(tid, "succeeded")

# %% CER online
def run_cer_online():
    task_id_list = parse_task_ids(args.task_ids)
    cer_path = cer_buffer_path(args.website, skill_tag("cer_online", args.model))

    for tid in tqdm(task_id_list, desc="cer_online", unit="task"):
        if args.resume and is_task_completed(tid, "cer_online"):
            print(f"[webarena.{tid}] already completed, skipping")
            tracker.mark(tid, "skipped")
            continue

        # Solve.
        if not run_step([
            "python", "run_demo.py",
            "--task_name", f"webarena.{tid}",
            "--model_name", args.model,
            "--results_dir", args.results_dir,
            "--websites", args.website,
            "--cer_store_path", cer_path,
            "--cer_top_k_dynamics", str(args.cer_top_k_dynamics),
            "--cer_top_k_skills", str(args.cer_top_k_skills),
            "--max_steps", "30",
            "--headless",
        ], "solve", tid, timeout=SOLVE_TIMEOUT):
            tracker.mark(tid, "failed")
            continue

        # Evaluate.
        run_step([
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--model", args.eval_model,
        ], "autoeval", tid)

        # Distill.
        if not run_step([
            "python", "-m", "induce.induce_cer_experience",
            "--model", args.model,
            "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--config_dir", "config_files",
            "--cer_store_path", cer_path,
        ], "induce_cer_experience", tid, timeout=240):
            print(f"[webarena.{tid}] cer distillation failed; "
                  f"task counted as succeeded, buffer unchanged")

        tracker.mark(tid, "succeeded")


# %% SGDR (state-grounded dynamic retrieval)
def run_sgdr():
    """Run SGDR online learning."""
    task_id_list = parse_task_ids(args.task_ids)
    skill_path = sgdr_skill_path(args.website, skill_tag("sgdr", args.model))
    # Activation logs.
    activation_log_dir = os.path.join(args.results_dir, "sgdr_logs")
    os.makedirs(activation_log_dir, exist_ok=True)

    eval_model = args.eval_model

    for tid in tqdm(task_id_list, desc="sgdr", unit="task"):
        if args.resume and is_task_completed(tid, "sgdr"):
            print(f"[webarena.{tid}] already completed, skipping")
            tracker.mark(tid, "skipped")
            continue

        # Solve.
        env = os.environ.copy()
        env["SGDR_ACTIVATION_LOG"] = os.path.join(activation_log_dir, f"{tid}.jsonl")
        if not run_step([
            "python", "run_demo.py",
            "--task_name", f"webarena.{tid}",
            "--model_name", args.model,
            "--results_dir", args.results_dir,
            "--websites", args.website,
            "--skill_store_path", skill_path,
            "--top_k", str(args.top_k),
            *(["--top_m", str(args.top_m)] if args.top_m is not None else []),
            "--alpha", str(args.alpha),
            "--mmr_lambda", str(args.mmr_lambda),
            "--use_mmr", str(args.use_mmr),
            "--headless",
        ], "solve", tid, timeout=SOLVE_TIMEOUT, env=env):
            tracker.mark(tid, "failed")
            continue

        path = f"{args.results_dir}/webarena.{tid}/summary_info.json"
        data = load_json_safe(path, tid, "solve")
        if data is None:
            tracker.mark(tid, "failed")
            continue
        if data["n_steps"] < 3:
            tracker.mark(tid, "filtered_steps")
            continue

        # Evaluate.
        if not run_step([
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--model", eval_model,
        ], "autoeval", tid):
            tracker.mark(tid, "failed")
            continue
        eval_data = load_json_safe(autoeval_path(f"{args.results_dir}/webarena.{tid}"),
                                   tid, "autoeval")
        if eval_data is None:
            tracker.mark(tid, "failed")
            continue
        if not eval_data[0]["rm"]:
            tracker.mark(tid, "filtered_eval")
            continue

        # Clean.
        if not run_step([
            "python", "utils/calc_valid_steps.py",
            "--clean_and_store", "--result_dir", f"{args.results_dir}/webarena.{tid}",
            "--model", args.model,
        ], "clean", tid):
            tracker.mark(tid, "failed")
            continue

        # Induce skills.
        if not args.no_induce:
            verify_args = []
            induce_timeout = 200
            induce_ok = run_step([
                "python", "-m", "induce.induce_skills_window",
                "--website", args.website,
                "--results_dir", args.results_dir,
                "--result_id_list", tid,
                "--model", eval_model,
                "--skill_store_path", skill_path,
                *verify_args,
            ], "induce_skills_window", tid, timeout=induce_timeout)
            if not induce_ok:
                print(f"[webarena.{tid}] WARNING: induction failed; "
                      f"task counted as succeeded but no skills were added.")

        tracker.mark(tid, "succeeded")


# %% Main Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["awm", "asi", "sgdr", "cer_online"])
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "admin", "reddit", "gitlab", "map"])
    parser.add_argument("--task_ids", type=str, required=True,
                        help="xxx-xxx,xxx-xxx")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Backbone model for the agent.")
    parser.add_argument("--eval_model", type=str, default=None,
                        help="Model for trajectory evaluation and induction. "
                             "Defaults to --model.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory. Defaults to 'results_<sanitized_model>'.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks that already have autoeval results.")

    # SGDR-specific knobs (ignored by other experiments).
    parser.add_argument("--top_k", type=int, default=5,
                        help="SGDR: top-K skills to activate per step.")
    parser.add_argument("--top_m", type=int, default=None,
                        help="SGDR: TopM candidate pool size for MMR rerank.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="SGDR: task-similarity weight (1-alpha to state).")
    parser.add_argument("--mmr_lambda", type=float, default=0.7,
                        help="SGDR: MMR weight on relevance vs. diversity.")
    parser.add_argument("--use_mmr", type=str, default="True",
                        help="SGDR: 'True' / 'False' — enable MMR rerank.")
    parser.add_argument("--reuse_skill_lib", action="store_true",
                        help="SGDR: keep the existing JSONL library instead of "
                             "archiving and seeding blank. Required for A/B reruns "
                             "against a frozen library.")
    parser.add_argument("--no_induce", action="store_true",
                        help="SGDR: skip the induction step after each task. "
                             "Pair with --reuse_skill_lib to freeze the library.")
    parser.add_argument("--cer_top_k_dynamics", type=int, default=5,
                        help="cer_online: top-K dynamics replayed per task.")
    parser.add_argument("--cer_top_k_skills", type=int, default=5,
                        help="cer_online: top-K skills replayed per task.")

    args = parser.parse_args()

    # vLLM override.
    effective_model = os.environ.get("LLM_MODEL_NAME", args.model)
    args.model = effective_model
    args.eval_model = os.environ.get("LLM_MODEL_NAME", args.eval_model) or args.model
    if args.alpha is None:
        args.alpha = 0.5

    if args.results_dir is None:
        args.results_dir = f"results/{args.experiment}_{sanitize_model_name(args.model)}"

    if "qwen" in args.model.lower():
        SOLVE_TIMEOUT = 600

    # Validate website.
    task_id_list = parse_task_ids(args.task_ids)
    first_tid = task_id_list[0]
    config_path = os.path.join("config_files", f"{first_tid}.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
        sites = config.get("sites", [])
        # Admin alias.
        website_aliases = {args.website, f"shopping_{args.website}" if args.website == "admin" else None} - {None}
        assert any(s in website_aliases for s in sites), (
            f"--website '{args.website}' does not match task {first_tid}'s sites {sites}. "
            f"Check your --task_ids or --website argument."
        )

    # Prepare run files.
    tag = skill_tag(args.experiment, args.model)
    if args.experiment == "sgdr":
        setup_sgdr_files(
            args.website,
            tag,
            reuse=args.reuse_skill_lib,
            label=args.experiment,
        )
    elif args.experiment == "cer_online":
        setup_cer_files(args.website, tag, reuse=args.resume)
    else:
        setup_skill_files(args.website, tag)

    if args.experiment == "awm":
        run_awm()
    elif args.experiment == "asi":
        run_asi()
    elif args.experiment == "sgdr":
        run_sgdr()
    elif args.experiment == "cer_online":
        run_cer_online()

    tracker.summary()
