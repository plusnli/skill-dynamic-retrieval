"""Induce Actions on the Full Task Level."""

import os
import gzip
import json
import pickle
import shutil
import argparse
from llm_client import llm_completion
import subprocess
from induce.utils import (
    extract_code_pieces, get_task_id, get_result_dirs,
    get_output_dir
)

# %% Induce Actions

def get_example_query_filtered(index: int, result_dir: str, config_dir: str) -> tuple[str, str]:
    """Build a filtered example."""
    cid = get_task_id(result_dir)
    config_path = os.path.join(config_dir, f"{cid}.json")
    config = json.load(open(config_path))
    instruction = config["intent"]
    task = config["intent_template"]

    step_dirs = [f for f in os.listdir(result_dir) if f.startswith("step") and f.endswith(".pkl.gz")]
    step_dirs = sorted(step_dirs, key=lambda x: int(x.split('.')[0].split('_')[1]))
    step_dirs = [os.path.join(result_dir, sd) for sd in step_dirs]
    steps = []
    for sd in step_dirs:
        step_info = pickle.load(gzip.open(sd, 'rb'))
        error_msg = step_info.obs["last_action_error"]
        if len(error_msg) > 0 and (not error_msg.startswith("TimeoutError")):
            continue
        if len(step_info.obs["last_action"]) == 0:
            continue
        steps.append(step_info.obs["last_action"])

    if len(steps) > 0:
        print(f"Collected #{len(steps)} valid steps from a total of #{len(step_dirs)} steps.")
        steps = '\n'.join(steps)
        ex = f"### Example {index} ({config['task_id']}): {instruction}\n{steps}"
    else:
        ex = None
    return ex, task


def get_example_query_cleaned(index: int, result_dir: str, config_dir: str) -> str:
    """Build a cleaned example."""
    # Config.
    cid = get_task_id(result_dir)
    config_path = os.path.join(config_dir, f"{cid}.json")
    config = json.load(open(config_path))

    # Instruction.
    subtask_inst_path = os.path.join(result_dir, "instruction.txt")
    if os.path.exists(subtask_inst_path):
        instruction = open(subtask_inst_path, 'r').read()
        task = instruction
    else:
        instruction = config["intent"]
        task = config["intent_template"]

    steps = json.load(open(os.path.join(result_dir, "cleaned_steps.json")))
    if len(steps) > 0:
        print(f"Collected #{len(steps)} valid steps.")
        steps = '\n'.join(steps)
        ex = f"### Example {index} ({config['task_id']}): {instruction}\n{steps}"
    else:
        ex = None

    return ex, task

def get_test_query(result_dir_list: str, config_dir: str) -> str:
    """Build the induction query."""
    task = None
    examples = []
    for rdir in result_dir_list:
        ex, task = get_example_query_cleaned(len(examples)+1, rdir, config_dir)
        if ex is None: continue
        examples.append(ex)
    
    if len(examples) < 1:
        return None
    query = f"## Task: {task}\n" + '\n\n'.join(examples)
    return query


def induce_actions() -> list[str] | None:
    result_dir_list = get_result_dirs(args.results_dir, args.result_id_list, args.template_id, args.config_dir)
    test_query = get_test_query(result_dir_list, args.config_dir)
    if test_query is None: return []
    with open(args.test_query_path, 'w') as fw: fw.write(test_query)

    messages = [{"role": "system", "content": open(args.sys_msg_path).read()}]
    messages += [{"role": "user", "content": open(args.instruction_path).read()}]
    messages += [{"role": "user", "content": open(args.few_shot_path).read()}]
    messages += [{"role": "user", "content": test_query + '\n\n## Reusable Functions'}]

    all_responses = []
    response = llm_completion(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        n=args.num_responses,
    )
    for i, resp in enumerate(response.choices):
        curr_resp = resp.message.content
        curr_path = os.path.join(args.output_dir, f"{i}.md")
        with open(curr_path, 'w') as fw:
            fw.write(test_query + '\n\n\n' + curr_resp)
        all_responses.append(curr_resp)
    return all_responses



# %% Process Actions
from induce.utils import count_function_calls, get_function_names

def write_actions(response: str) -> tuple[str, list[str]]:
    """Append induced actions."""
    existing_action_names = get_function_names(open(args.write_action_path, 'r').read())
    # Extract functions.
    actions = extract_code_pieces(response, start="```python", end="```", do_split=False)
    actions = [a for a in actions if "def " in a and count_function_calls(a, 1)]
    new_actions, action_names = [], []
    for a in actions:
        if ("def " in a) and count_function_calls(a, 1):
            a_names = get_function_names(a, existing_action_names)
            if len(a_names) > 0:
                action_names.extend(a_names)
                new_actions.append(a)

    print(
        f"Induced #{len(new_actions)}|{len(action_names)} Actions, ",
        [a.split("\n")[0] for a in new_actions],
        action_names
    )
    if len(new_actions) == 0: return None, None

    tmp_path = args.write_action_path + ".tmp"
    process = subprocess.Popen(["cp", args.write_action_path, tmp_path])
    process.wait()

    with open(args.write_action_path, 'a+') as fw:
        fw.write('\n\n'+ '\n\n'.join(new_actions))
    return tmp_path, action_names



# %% Run Tests
from induce.utils import parse_tests

def write_tests(response: str, result_id_list: list[str], action_names: list[str] = []) -> bool:
    """Write and run induced-action tests."""
    tests = parse_tests(response, action_names)
    assert len(tests) == len(result_id_list), f"Got #{len(tests)} tests but for #{len(result_id_list)} results."
    
    # Write test script.
    os.makedirs(args.write_tests_dir, exist_ok=True)
    script_content = []
    for i, (t, r) in enumerate(zip(tests, result_id_list)):
        # Test trajectory.
        test_path = os.path.join(args.write_tests_dir, f"test_{i}.txt")
        test_str = '\n'.join([f"```{tl.strip()}```" for tl in t.split('\n') if tl.strip()])
        with open(test_path, 'w') as fw:
            fw.write(test_str)
        
        # Test command.
        script_content.append(f"# Run test for task {r}")
        script_content.append(
            f"python run_demo.py --websites {args.website} --headless "
            f"--task_name webarena.{r.split('_')[0]} "
            f"--action_path {test_path} "
            f"--results_dir {args.results_dir} "
            f"--rename_to webarena.{r.split('_')[0]}_test",
        )
        script_content.append("\n")
    
    test_script_path = os.path.join(args.write_tests_dir, "run_tests.sh")
    with open(test_script_path, 'w') as fw:
        fw.write('\n'.join(script_content))
    
    # Run tests.
    process = subprocess.Popen(["bash", test_script_path])
    try:
        stdout, stderr = process.communicate(timeout=100)
        print(stdout)
    except subprocess.TimeoutExpired as e:
        process.kill()
        stdout, stderr = process.communicate()
        print(f"Process timed out after {e.timeout} seconds.")
        print(stderr)
        return True

    # Check results.
    scores = []
    for r in result_id_list:
        if args.eval_with_gold:
            eval_path = os.path.join(args.results_dir, f"webarena.{r}_test", "summary_info.json")
            if os.path.exists(eval_path):
                scores.append(json.load(open(eval_path))["cum_reward"] == 1.0)
            else:
                scores.append(False)
        else:
            test_dir = os.path.join(args.results_dir, f"webarena.{r}_test")
            if not os.path.exists(test_dir):
                print(f"Test dir not found (run_demo likely failed): {test_dir}")
                scores.append(False)
            else:
                process = subprocess.Popen([
                    "python", "-m", "autoeval.evaluate_trajectory",
                    "--result_dir", test_dir,
                ])
                process.wait()

                eval_path = None
                for f in os.listdir(test_dir):
                    if f.endswith("_autoeval.json"):
                        eval_path = os.path.join(test_dir, f)
                        break
                if eval_path and os.path.exists(eval_path):
                    scores.append(json.load(open(eval_path))[0]["rm"] == True)
                else:
                    scores.append(False)

        # Check action use.
        test_dir_check = os.path.join(args.results_dir, f"webarena.{r}_test")
        if not os.path.exists(test_dir_check):
            print(f"Validity Check: skipped (test dir missing)")
            scores[-1] = False
        else:
            command = [
                "python", "utils/calc_valid_steps.py",
                "--result_dir", test_dir_check,
                "--action_names"] + action_names
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            stdout, _ = process.communicate()
            output = stdout.decode("utf-8").strip()
            output = output.split('\n')[-1].strip()
            print("Validity Check: ", output)
            if output == 'False': scores[-1] = False

        if scores[-1] == False: break
    
    print("Scores: ", scores)
    if all([s == True for s in scores]):
        print("All Tests Passed!")
        return False
    else:
        return True


# %% Overall pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--num_responses", type=int, default=1, help="Number of responses to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")

    parser.add_argument("--sys_msg_path", type=str, default="induce/prompt/system_message.txt")
    parser.add_argument("--instruction_path", type=str, default="induce/prompt/instruction.txt")
    parser.add_argument("--few_shot_path", type=str, default="induce/prompt/shopping.md")
    parser.add_argument("--test_query_path", type=str, default="induce/prompt/test_query.txt")

    parser.add_argument("--template_id", type=str, default=None)
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "admin", "reddit", "gitlab", "map"])
    parser.add_argument("--config_dir", type=str, default="config_files")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--result_id_list", type=str, nargs="+", default=None, help="E.g., '110 111'.")

    parser.add_argument("--write_action_path", type=str, default=None)
    parser.add_argument("--write_tests_dir", type=str, default="debug_actions")
    parser.add_argument("--outputs_root", type=str, default=None, required=True,
                        help="Experiment-tagged root for the LLM-response cache, e.g. "
                             "outputs/<experiment>_<model> (see get_output_dir). REQUIRED: "
                             "the bare flat root 'outputs' is rejected because it silently "
                             "shares induced skills across methods/models that run on the "
                             "same (website, task_ids).")
    parser.add_argument("--eval_with_gold", action="store_true", help="If perform evaluation with ground-truth.")
    args = parser.parse_args()

    # Action output.
    if args.write_action_path is None:
        args.write_action_path = os.path.join("actions", f"{args.website}.py")

    # Cache path.
    args = get_output_dir(args)
    if os.path.exists(args.output_dir):
        print(f"Output directory already exists: {args.output_dir}")
        names = sorted(os.listdir(args.output_dir), key=lambda x: int(x.split('.')[0]))
        paths = [os.path.join(args.output_dir, f) for f in names]
        responses = [open(p, 'r').read() for p in paths]
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        responses = induce_actions()
    
    # Evaluate responses.
    print(f"Collected {len(responses)} Responses..")
    for resp_idx, resp in enumerate(responses):
        print(f"\n\n** Start Evaluating Response {resp_idx}: ", resp, "**")
        tmp_path, action_names = write_actions(resp)
        if tmp_path is None: continue

        if_revert = write_tests(resp, args.result_id_list, action_names)
        print("If Revert: ", if_revert)
        if if_revert:
            # Preserve symlink binding.
            shutil.copyfile(tmp_path, args.write_action_path)
            os.remove(tmp_path)
            for j, r in enumerate(args.result_id_list):
                print("Command: ", ["rm", "-rf", f"results/webarena.{r}_test"])
                process = subprocess.Popen(["rm", "-rf", f"results/webarena.{r}_test"])
                process.wait()
        else:
            process = subprocess.Popen(["rm", tmp_path])
            process.wait()
            break
            
        print(f"** Finish Evaluating Response {resp_idx}: ", resp, "**\n\n")
        
