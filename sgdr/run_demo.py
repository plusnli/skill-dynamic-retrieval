import os
import argparse
import warnings
from datetime import datetime

# LiteLLM cleanup warning.
warnings.filterwarnings("ignore", message="coroutine 'close_litellm_async_clients' was never awaited")

from agent import DemoAgentArgs
from patch_with_custom_exec import patch_with_custom_exec

from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4.1",
        help="Model name. For commercial APIs: 'gpt-4o-mini', 'gpt-4o', etc. "
             "For local vLLM: use the model name as loaded (overridden by LLM_MODEL_NAME env var).",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )

    parser.add_argument(
        "--websites", type=str, nargs='+', default=[],
        choices=["shopping", "admin", "reddit", "gitlab", "map"],
        help="Name of the website(s) to run the agent on. Used to define agent's action space.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=10,
        help="Maximum number of steps to run the agent.",
    )
    
    # Debug paths.
    parser.add_argument(
        "--action_path", type=str, default=None, # "debug_actions/test.txt",
        help="Path to the specified actions for agents to take.",
    )
    parser.add_argument(
        "--memory_path", type=str, default=None, # "memory/test.txt",
        help="Path to the workflow memory.",
    )
    parser.add_argument(
        "--rename_to", type=str, default=None,
        help="If specified, rename the experiment folder to the specified name.",
    )
    parser.add_argument("--headless", action="store_true", help="Run the browser in headless mode.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to store experiment results. "
                             "Defaults to 'results_<model_name>' (/ replaced with _).")

    # SGDR retrieval.
    parser.add_argument("--skill_store_path", type=str, default=None,
                        help="Path to an SGDR JSONL skill library. "
                             "When set, the agent retrieves top-K skills per step.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="SGDR: top-K skills to activate per step.")
    parser.add_argument("--top_m", type=int, default=None,
                        help="SGDR: TopM candidate pool size for MMR rerank "
                             "(spec §5 R_{i,k}). Defaults to 3*top_k when unset.")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="SGDR: task-similarity weight (1-alpha goes to state).")
    parser.add_argument("--mmr_lambda", type=float, default=0.7,
                        help="SGDR: MMR weight on relevance vs. diversity.")
    parser.add_argument("--use_mmr", type=str2bool, default=True,
                        help="SGDR: enable MMR rerank in top-K.")
    parser.add_argument("--summarizer_model", type=str, default=None,
                        help="SGDR: model for the state summarizer (defaults to --model_name).")
    parser.add_argument("--cer_store_path", type=str, default=None,
                        help="CER: path to the dynamic replay buffer JSON.")
    parser.add_argument("--cer_top_k_dynamics", type=int, default=5,
                        help="CER: retrieve top-k dynamics per task.")
    parser.add_argument("--cer_top_k_skills", type=int, default=5,
                        help="CER: retrieve top-k skills per task.")

    return parser.parse_args()


def main():
    print(
        """\
--- WARNING ---
This is a basic agent for demo purposes.
Visit AgentLab for more capable agents with advanced features.
https://github.com/ServiceNow/AgentLab"""
    )

    args = parse_args()
    # vLLM override.
    effective_model = os.environ.get("LLM_MODEL_NAME", args.model_name)
    args.model_name = effective_model
    if args.results_dir is None:
        args.results_dir = f"results/demo_{effective_model.replace('/', '_')}"
    if args.rename_to is None:
        args.rename_to = args.task_name

    if args.action_path is not None and os.path.exists(args.action_path):
        actions = open(args.action_path, 'r').read()
        if actions.strip():
            actions = actions.splitlines()
        else:
            actions = []
    else:
        actions = []
    # Agent config.
    agent_args = DemoAgentArgs(
        model_name=args.model_name,
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
        websites=args.websites,
        actions=tuple(actions),
        memory=args.memory_path,
        skill_store_path=args.skill_store_path,
        cer_store_path=args.cer_store_path,
        cer_top_k_dynamics=args.cer_top_k_dynamics,
        cer_top_k_skills=args.cer_top_k_skills,
        top_k=args.top_k,
        top_m=args.top_m,
        alpha=args.alpha,
        mmr_lambda=args.mmr_lambda,
        use_mmr=args.use_mmr,
        summarizer_model=args.summarizer_model,
    )
    
    patch_with_custom_exec(agent_args)

    # Environment config.
    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        # viewport={"width": 1500, "height": 1280},
    )

    # Open-ended task.
    if args.task_name == "openended":
        agent_args.chat_mode = True
        env_args.wait_for_user_message = False # True
        env_args.task_kwargs = {"start_url": args.start_url}

    # Experiment config.
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    # Run experiment.
    exp_args.prepare(args.results_dir)
    exp_args.run()

    # Print results.
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")

    if args.rename_to is not None:
        target_dir = os.path.join(args.results_dir, args.rename_to)
        source_dir = exp_args.exp_dir

        # Preserve canonical result name.
        if os.path.abspath(source_dir) != os.path.abspath(target_dir):
            if os.path.exists(target_dir):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = f"{target_dir}__bak_{ts}"
                suffix = 1
                while os.path.exists(backup_dir):
                    backup_dir = f"{target_dir}__bak_{ts}_{suffix}"
                    suffix += 1
                os.rename(target_dir, backup_dir)
                print(f"Archived existing result dir to: {backup_dir}")
            os.rename(source_dir, target_dir)


if __name__ == "__main__":
    main()
