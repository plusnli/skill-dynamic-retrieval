import os
import gzip
import glob
import json
import pickle
import argparse
import traceback
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT


def load_final_state(result_dir: str) -> str:
    """Load the final accessibility tree."""
    step_paths = glob.glob(os.path.join(result_dir, "step_*.pkl.gz"))
    step_paths = sorted(
        step_paths,
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    for path in reversed(step_paths):
        try:
            step = pickle.load(gzip.open(path, "rb"))
        except Exception:
            continue
        obs = getattr(step, "obs", None)
        if not obs:
            continue
        axtree = obs.get("axtree_txt")
        if axtree and axtree.strip():
            return axtree
    return ""


def load_blocks(path: str) -> list[str]:
    blocks, curr_block = [], []
    for line in open(path, 'r'):
        if "INFO" in line:
            if curr_block:
                blocks.append(curr_block)
                curr_block = []
        else:
            curr_block.append(line.strip())
    if curr_block:
        blocks.append(curr_block)
    blocks = ['\n'.join(b) for b in blocks]
    blocks = [b for b in blocks if b.strip()]
    return blocks


def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid actions."""
    import ast
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            if type(ast.literal_eval(arg)) == str:
                valid_actions.append(a)
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(ast.literal_eval(arg)) == str:
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions


def extract_code_pieces(text: str) -> list[str]:
    """Extract fenced code blocks."""
    code_pieces = []
    while "```" in text:
        st_idx = text.index("```") + len("```")
        if "```" in text[st_idx:]:
            end_idx = text.index("```", st_idx + 1)
        else: 
            end_idx = len(text)
        code_pieces.append(text[st_idx:end_idx].strip())
        text = text[end_idx+3:].strip()
    return code_pieces


def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    blocks = load_blocks(path)
    think_list, action_list = [], []
    for b in blocks:
        if '```' in b:
            sidx = b.index('```')
            think_list.append(b[:sidx].strip())
            actions = extract_code_pieces(b[sidx:])
            action_list.extend(actions)
        else:
            # Plain code fallback.
            think_list.append("")
            for line in b.split("\n"):
                line = line.strip()
                if line:
                    action_list.append(line)
    return think_list, action_list


def extract_response(action: str) -> str:
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]


def process_sample(
    idx: str, traj_info: dict, log_save_path,
    model: str, eval_version: str,
) -> list[dict]:
    # Select judge context.
    clients = {model: CLIENT_DICT[model](model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, _ = evaluator(traj_info, model, eval_version)
        eval_result = None
        if out["status"].lower() == "success": eval_result = True
        else: eval_result = False
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"], 
                "uid": traj_info["traj_name"],
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return [{
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None,
            "uid": traj_info["traj_name"],
        }]


def main():
    # Task config.
    task_id = args.result_dir.split('/')[-1].split(".")[1]
    if '_' in task_id:
        task_id = task_id.split('_')[0]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))

    # Trajectory log.
    log_path = os.path.join(args.result_dir, "experiment.log")
    think_list, action_list = extract_think_and_action(log_path)
    if "send_msg_to_user" in action_list[-1]:
        response = extract_response(action_list[-1])
    else:
        response = ""
    
    # Summary info.
    summary_path = os.path.join(args.result_dir, "summary_info.json")
    summary = json.load(open(summary_path, 'r'))

    # Trajectory info.
    image_paths = [
        os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir) 
        if f.startswith("screenshot_step_") and f.endswith(".png")
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))

    subtask_dir = args.result_dir.removesuffix("_test")
    inst_path = os.path.join(subtask_dir, "instruction.txt")
    if os.path.exists(inst_path):
        intent = open(inst_path, 'r').read().strip()
    else:
        intent = config["intent"]
    print("\n\n", "="*50)
    print("Intent:", intent)
    print("="*50, "\n\n")
    # Final state.
    final_state = load_final_state(args.result_dir) if args.eval_source == "state" else ""
    traj_info = {
        "intent": intent,
        "response": response,
        "final_state": final_state,
        "eval_source": args.eval_source,
        "captions": think_list,
        "actions": action_list,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"]
    }

    # Evaluate.
    log_save_path = os.path.join("autoeval/log", args.result_dir.split('/')[-1])
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path, 
        model=args.model, eval_version=args.prompt,
    )
    safe_model_name = args.model.replace("/", "_")
    output_eval_path = os.path.join(args.result_dir, f"{safe_model_name}_autoeval.json")
    json.dump(eval_info, open(output_eval_path, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the result directory, e.g., 'webarena.0'.")
    parser.add_argument("--model", type=str, default="gpt-4.1",
                        help="Eval model. For commercial: 'gpt-4o', 'gpt-4o-mini', etc. "
                             "For local vLLM: use the loaded model name.")
    parser.add_argument("--prompt", type=str, default=None,
                        choices=["text", "vision"],
                        help="Eval prompt type. Defaults to 'text'. Pass --prompt vision explicitly to enable image input.")
    parser.add_argument("--eval_source", type=str, default="state",
                        choices=["state", "think"],
                        help="Which page info the text judge sees in the "
                             "'detailed final state of the webpage' slot: "
                             "'state' = final-page accessibility tree (default), "
                             "'think' = agent's last-step thought (legacy behavior).")

    args = parser.parse_args()

    # Prompt mode.
    if args.prompt is None:
        args.prompt = "text"

    main()
