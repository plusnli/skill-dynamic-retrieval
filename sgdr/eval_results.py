"""Evaluate task success rates."""

import os
import json
import math
import argparse

# Prefix subsets.
SUBSET_PCT = {"P10": 10, "P30": 30, "P50": 50}


def subset_ids(task_ids: list[int], subset: str) -> list[int]:
    """Return a prefix subset."""
    if subset == "full":
        return task_ids
    pct = SUBSET_PCT[subset]
    cut = max(1, math.ceil(len(task_ids) * pct / 100))
    return task_ids[:cut]

# Task IDs.
TASK_IDS = {
    "shopping":  "21-26,47-51,96,117-118,124-126,141-150,158-167,188-192,225-235,238-242,260-264,269-286,298-302,313,319-338,351-355,358-362,368,376,384-388,431-440,465-469,506-521,528-532,571-575,585-589,653-657,689-693,792-798",
    "admin":     "0-6,11-15,41-43,62-65,77-79,94-95,107-116,119-123,127-131,157,183-187,193-204,208-217,243-247,288-292,344-348,374-375,423,453-464,470-474,486-505,538-551,676-680,694-713,768-782,790",
    "reddit":    "27-31,66-69,399-410,580-584,595-652,714-735",
    "gitlab":    "44-46,102-106,132-136,156,168-182,205-207,258-259,293-297,303-312,314-318,339-343,349-350,357,389-398,411-422,441-452,475-485,522-527,533-537,567-570,576-579,590-594,658-670,736,742-756,783-789,799-811",
    "map":       "7-10,16-20,32-40,52-61,70-76,80-93,98-101,137-140,151-155,218-224,236-237,248-257,287,356,363-367,369-373,377-383,757-758,761-767",
}


def parse_ids(s: str) -> list[int]:
    ids = []
    for chunk in s.split(","):
        parts = chunk.strip().split("-")
        if len(parts) == 1:
            ids.append(int(parts[0]))
        else:
            ids.extend(range(int(parts[0]), int(parts[1]) + 1))
    return ids


def find_autoeval_file(result_dir: str) -> str | None:
    """Find autoeval JSON."""
    if not os.path.isdir(result_dir):
        return None
    for f in os.listdir(result_dir):
        if f.endswith("_autoeval.json"):
            return os.path.join(result_dir, f)
    return None


def eval_task(result_dir: str, metric: str) -> bool | None:
    """Evaluate one task."""
    summary_path = os.path.join(result_dir, "summary_info.json")
    autoeval_path = find_autoeval_file(result_dir)

    env_result = None
    if os.path.exists(summary_path):
        try:
            data = json.load(open(summary_path))
            env_result = data.get("cum_reward", 0) > 0
        except (json.JSONDecodeError, KeyError):
            pass

    autoeval_result = None
    if autoeval_path:
        try:
            data = json.load(open(autoeval_path))
            autoeval_result = data[0].get("rm") == True
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    if metric == "env":
        return env_result
    elif metric == "autoeval":
        return autoeval_result
    elif metric == "either":
        if env_result is None and autoeval_result is None:
            return None
        return bool(env_result) or bool(autoeval_result)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_website(results_dir: str, website: str, metric: str, subset: str = "full") -> dict:
    task_ids = subset_ids(parse_ids(TASK_IDS[website]), subset)
    total = len(task_ids)
    completed = 0
    success = 0
    missing = []

    for tid in task_ids:
        result_dir = os.path.join(results_dir, f"webarena.{tid}")
        result = eval_task(result_dir, metric)
        if result is None:
            missing.append(tid)
        else:
            completed += 1
            if result:
                success += 1

    return {
        "website": website,
        "total": total,
        "completed": completed,
        "missing": len(missing),
        "success": success,
        "rate_of_total": success / total if total > 0 else 0.0,
        "rate_of_completed": success / completed if completed > 0 else 0.0,
    }


def print_table(rows: list[dict], metric: str):
    print(f"\nMetric: {metric}")
    print(f"{'Website':<12} {'Total':>6} {'Done':>6} {'Missing':>8} {'Success':>8} {'Rate(all)':>10} {'Rate(done)':>11}")
    print("-" * 65)

    grand_total = grand_done = grand_success = 0
    for r in rows:
        grand_total   += r["total"]
        grand_done    += r["completed"]
        grand_success += r["success"]
        rate_all  = f"{r['rate_of_total']*100:.1f}%"
        rate_done = f"{r['rate_of_completed']*100:.1f}%" if r["completed"] > 0 else "  N/A"
        status = "" if r["missing"] == 0 else f" ({r['missing']} missing)"
        print(f"{r['website']:<12} {r['total']:>6} {r['completed']:>6} {r['missing']:>8} {r['success']:>8} {rate_all:>10} {rate_done:>11}{status}")

    print("-" * 65)
    rate_all  = f"{grand_success/grand_total*100:.1f}%" if grand_total > 0 else "N/A"
    rate_done = f"{grand_success/grand_done*100:.1f}%"  if grand_done  > 0 else "N/A"
    print(f"{'TOTAL':<12} {grand_total:>6} {grand_done:>6} {grand_total-grand_done:>8} {grand_success:>8} {rate_all:>10} {rate_done:>11}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["awm", "asi", "sgdr", "cer_online"],
                        help="Experiment mode.")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Model name used in the run (determines results_dir).")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results directory (default: results/{experiment}_{model}).")
    parser.add_argument("--websites", type=str, nargs="+",
                        choices=list(TASK_IDS.keys()), default=list(TASK_IDS.keys()),
                        help="Websites to evaluate (default: all).")
    parser.add_argument("--metric", type=str, default="env",
                        choices=["env", "autoeval", "either"],
                        help="env=cum_reward>0, autoeval=LLM-judge, either=union (default: env).")
    parser.add_argument("--subset", type=str, default="full",
                        choices=["full", "P10", "P30", "P50"],
                        help="Evaluate only the first 10/30/50%% of each website's "
                             "task list (matches task_subsets.sh). Default: full.")
    args = parser.parse_args()

    model_safe = args.model.replace("/", "_")
    results_dir = args.results_dir or f"results/{args.experiment}_{model_safe}"

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        exit(1)

    print(f"Results dir : {results_dir}")
    print(f"Websites    : {', '.join(args.websites)}")
    print(f"Subset      : {args.subset}")

    rows = []
    for website in args.websites:
        if website not in TASK_IDS:
            print(f"Warning: unknown website '{website}', skipping.")
            continue
        rows.append(evaluate_website(results_dir, website, args.metric, args.subset))

    print_table(rows, args.metric)
