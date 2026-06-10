"""Validate and clean trajectory steps."""

import os
import re
import gzip
import json
import pickle
import argparse
from llm_client import llm_completion

def is_valid_step(step_info) -> bool:
    """Check action validity."""
    if step_info.obs is None: return False
    error_msg = step_info.obs["last_action_error"]
    if len(error_msg) > 0 and (not error_msg.startswith("TimeoutError")):
        return False
    if len(step_info.obs["last_action"]) == 0:
        return False
    return True


INFO_ACTIONS = ["send_msg_to_user", "report_infeasible"]
def is_state_change_step(step_info, last_step_info) -> bool:
    """Check state change."""
    if step_info.obs is None: return False
    # User message.
    if any([ia in step_info.obs["last_action"] for ia in INFO_ACTIONS]):
        return True
    # Initial step.
    if last_step_info is None:
        return False
    return step_info.obs["axtree_txt"] != last_step_info.obs["axtree_txt"]


def has_induced_actions(step_info, action_names: set) -> bool:
    if step_info.obs is None: return False
    return any([a in step_info.obs["last_action"] for a in action_names])


SIMPLIFY_THOUGHT_INSTRUCTION = """You are a helpful assistant in summarizing web browsing actions. Your task is to summarize the main action taken and the corresponding state change, from the input thought process and environmental description. 

Example input: 
'''Let me think about this step by step:
1) I've clicked the Columns button and based on the accessibility tree, I can now see a list of available columns that can be shown/hidden.
2) Looking through the available columns in the accessibility tree, I don't see any direct column for "Number of Orders" or similar metric.
3) Since I don't see the information I need in the available columns, this suggests that:
   - Either this information is available through a different path
   - Or we need to access the Sales/Orders section to correlate this information
4) The most logical path would be to:
   - Go to the Sales section where we can see all orders
   - And then analyze which customers have the most orders
5) From the accessibility tree, I can see the "SALES" link with bid "183" in the main navigation.

Let me click on the SALES link:
This should open the Sales menu where we can access the Orders section to analyze customer order histories.'''
 
Example output:
'''Clicked the SALES link to access the Orders section for customer order histories.'''
"""

def simplify_thought(step: str) -> str:
    """Simplify the thought part of a step."""
    code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    code_parts = code_pattern.findall(step)
    text_parts = code_pattern.split(step)
    # Non-code text.
    text_parts = [part.strip() for i, part in enumerate(text_parts) if i % 2 == 0 and part.strip()]

    if len(text_parts) > 0:
        messages = [
            {"role": "system", "content": SIMPLIFY_THOUGHT_INSTRUCTION},
            {"role": "user", "content": '\n'.join(text_parts)},
        ]
        response = llm_completion(
            model=args.model,
            messages=messages,
            temperature=0.0,
        )
        simplified_thought = response.choices[0].message.content
        simplified_step = f"{simplified_thought}\n"
    else:
        simplified_step = ""
    if len(code_parts) == 0:
        # Plain code.
        simplified_step += f"```{step.strip()}```"
    elif len(code_parts) > 1:
        code = '\n'.join(code_parts)
        simplified_step += f"```\n{code}\n```"
    else:
        simplified_step += f"```{code_parts[0]}```"
    return simplified_step



def main():
    step_dirs = [f for f in os.listdir(args.result_dir) if f.startswith("step") and f.endswith(".pkl.gz")]
    step_dirs = sorted(step_dirs, key=lambda x: int(x.split('.')[0].split('_')[1]))
    step_dirs = [os.path.join(args.result_dir, sd) for sd in step_dirs]
    print(f"Found #{len(step_dirs)} steps in result dir {args.result_dir}.")

    is_valid, is_state_change, is_induction = [], [], []
    last_step_info = None
    for i, sd in enumerate(step_dirs):
        step_info = pickle.load(gzip.open(sd, 'rb'))
        is_valid.append(is_valid_step(step_info))
        is_state_change.append(is_state_change_step(step_info, last_step_info))
        is_induction.append(has_induced_actions(step_info, args.action_names))

        last_step_info = step_info

    is_valid, is_state_change, is_induction = is_valid[1: ], is_state_change[1: ], is_induction[1: ]
    print(f"Valid Steps ({sum(is_valid)}/{len(step_dirs)}): ", is_valid)
    print(f"State Change ({sum(is_state_change)}/{len(step_dirs)}): ", is_state_change)
    print(f"Action Induced ({sum(is_induction)}/{len(step_dirs)}): ", is_induction)

    # Check induced action usage.
    pass_checks = False
    for v, s, i in zip(is_valid, is_state_change, is_induction):
        if i:
            if (v and s): 
                pass_checks = True
                break
    print(str(pass_checks))

    if args.clean_and_store == True:
        cleaned_steps = []
        assert len(is_valid) == len(step_dirs[1: ])
        for valid, sd in zip(is_valid, step_dirs[1: ]):
            step_info = pickle.load(gzip.open(sd, 'rb'))
            if valid:
                simplified_step = simplify_thought(step_info.obs["last_action"])
                cleaned_steps.append(simplified_step)
        
        with open(os.path.join(args.result_dir, "cleaned_steps.json"), 'w') as fw:
            json.dump(cleaned_steps, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--action_names", type=str, nargs='+', default="")

    parser.add_argument("--clean_and_store", action="store_true", help="Clean and store the valid steps.")
    parser.add_argument("--model", type=str, default="gpt-4.1")

    args = parser.parse_args()

    main()
