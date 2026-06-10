import inspect
import logging
import os
import random
import browsergym.core.action.utils as utils
import pyparsing as pp
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

from typing import Optional
from browsergym.core.action.functions import noop
from browsergym.core.action.highlevel import (
    ACTION_SUBSETS,
    HighLevelActionSet,
)

from dataclasses import dataclass
@dataclass
class HighLevelAction:
    signature: str
    description: str
    examples: list[str]
    function: str
    add_code: bool

from parsers import _build_python_subset_parser

# Python action parser.
python_subset_parser: pp.ParserElement = _build_python_subset_parser()

class CustomActionSet(HighLevelActionSet):

    def __init__(
        self,
        subsets: Optional[HighLevelActionSet.ActionSubset | list[HighLevelActionSet.ActionSubset]] = [
            "chat",
            "infeas",
            "bid",
            "nav",
            "tab",
        ],
        custom_actions: Optional[list[callable]] = None,
        multiaction: bool = True,
        demo_mode: Optional[HighLevelActionSet.DemoMode] = None,
        strict: bool = False,
        retry_with_force: bool = False,
        retrievable_actions: Optional[list[str]] = None,
        retrieval_model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ):
        self.strict = strict
        self.multiaction = multiaction
        self.demo_mode = demo_mode
        self.retry_with_force = retry_with_force
        if retrievable_actions is None:
            retrievable_actions = []
        self.retrieval_model_name = retrieval_model_name

        if not subsets:
            raise ValueError(f"'action_subsets' is empty.")

        if isinstance(subsets, str):
            subsets = [subsets]

        allowed_actions = [noop]

        # Add actions.
        if subsets:
            for subset in subsets:
                if subset in ACTION_SUBSETS:
                    allowed_actions.extend(ACTION_SUBSETS[subset])
                elif subset == "custom":
                    if not custom_actions:
                        raise ValueError(
                            "'custom' is in 'action_subsets' but 'custom_actions' is empty."
                        )
                    allowed_actions.extend(custom_actions)
                else:
                    raise ValueError(f"Unknown high-level action subspace: {subset}")

        # Deduplicate in order.
        allowed_actions = list(dict.fromkeys(allowed_actions).keys())
        retrievable_actions = list(dict.fromkeys(retrievable_actions).keys())

        # Build action space.
        self.action_set: dict[str, HighLevelAction] = {}
        self.retrievable_action_set: dict[str, HighLevelAction] = {}
        self.python_includes = ""

        # Runtime imports.
        self.python_includes += f"""\
import playwright.sync_api
import requests
from typing import Literal
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from bs4 import BeautifulSoup


"""
        # Runtime flags.
        self.python_includes += f"""\
demo_mode={repr(demo_mode)}
retry_with_force={repr(retry_with_force)}

if demo_mode is None:
    demo_mode = "default" if DEMO_MODE else "off"

"""

        # Utility functions.
        for _, func in inspect.getmembers(utils, inspect.isfunction):
            self.python_includes += f"""\
{inspect.getsource(func)}


"""

        self._parse_and_include_actions(allowed_actions, self.action_set)
        self._parse_and_include_actions(retrievable_actions, self.retrievable_action_set)

        if retrievable_actions:
            self._build_retrieval_index()

        # Base snapshot.
        self._base_action_set: dict[str, HighLevelAction] = dict(self.action_set)
        self._base_python_includes: str = self.python_includes
        # Active SGDR skills.
        self._dynamic_skill_names: list[str] = []
    

    def _build_retrieval_index(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else self.device)
        self.retrieval_model = SentenceTransformer(self.retrieval_model_name, trust_remote_code=True).to(self.device)
        self.retrieval_max_length = 512
        action_descriptions = [self.get_action_doc(action, with_long_description=True, with_examples=True) for action in self.retrievable_action_set.values()]
        self.action_embeddings = self.retrieval_model.encode(action_descriptions, instruction="", max_length=self.retrieval_max_length, convert_to_tensor=True)
        
    def retrieve_actions(self, query: str, num_retrieve: int) -> dict[str, HighLevelAction]:
        instruction = "Instruct: Given a user query, retrieve functions that could be useful to fulfill the query.\nQuery: "
        query_embedding = self.retrieval_model.encode(query, instruction=instruction, max_length=self.retrieval_max_length, convert_to_tensor=True)
        scores = (query_embedding @ self.action_embeddings.T).squeeze()
        assert len(scores) == len(self.retrievable_action_set)
        top_actions = scores.argsort(descending=True)[:num_retrieve]
        return {list(self.retrievable_action_set.keys())[i]: list(self.retrievable_action_set.values())[i] for i in top_actions}
        
            
    def _parse_and_include_actions(self, allowed_actions, action_set):
        # Include actions.
        for func in allowed_actions:

            # Source code.
            self.python_includes += f"""\
{inspect.getsource(func)}


"""
            # Signature.
            signature = f"{func.__name__}{inspect.signature(func)}"

            # Docstring.
            description, examples = func.__doc__.split("Examples:", maxsplit=1)

            if func.__name__ in self.action_set:
                raise ValueError(f"Duplicated action '{func.__name__}'")

            action_file = inspect.getfile(func)
            actions_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "actions"))
            action_path = os.path.realpath(action_file)
            add_code = (
                os.path.commonpath([actions_root, action_path]) == actions_root
                and os.path.basename(action_file) != "__init__.py"
            )
            action_set[func.__name__] = HighLevelAction(
                signature=signature,
                description=description,
                examples=[examples.strip()],
                function=inspect.getsource(func),
                add_code=add_code,
            )
    
    # Dynamic skill layer.

    def set_dynamic_skills(self, skills) -> list[str]:
        """Replace current SGDR skills."""
        # Reset layer.
        self.action_set = dict(self._base_action_set)
        self.python_includes = self._base_python_includes
        self._dynamic_skill_names = []

        if not skills:
            return []

        injected: list[str] = []
        seen: set[str] = set()
        extra_includes: list[str] = []

        for skill in skills:
            name = getattr(skill, "func_name", None)
            code = getattr(skill, "code", None)
            description = getattr(skill, "description", "") or ""
            if not name or not code:
                continue
            if name in self._base_action_set or name in seen:
                logger.debug("[CustomActionSet] skip dynamic skill name collision: %s", name)
                continue

            ns: dict = {}
            try:
                exec(code, ns)
            except Exception as e:
                logger.warning("[CustomActionSet] failed to exec dynamic skill %s: %s", name, e)
                continue

            func = ns.get(name)
            if not callable(func):
                logger.warning("[CustomActionSet] dynamic skill %s: no callable named '%s' after exec", name, name)
                continue

            try:
                signature = f"{name}{inspect.signature(func)}"
            except (TypeError, ValueError):
                signature = f"{name}(...)"

            self.action_set[name] = HighLevelAction(
                signature=signature,
                description=description,
                examples=[],
                function=code.rstrip() + "\n",
                add_code=True,
            )
            extra_includes.append(code.rstrip() + "\n\n\n")
            seen.add(name)
            injected.append(name)

        if extra_includes:
            self.python_includes = self._base_python_includes + "\n" + "".join(extra_includes)
        self._dynamic_skill_names = list(injected)
        return injected

    def clear_dynamic_skills(self) -> None:
        """Shortcut for set_dynamic_skills([])."""
        self.set_dynamic_skills([])

    def example_action(self, abstract: bool, max_examples: int = 3) -> str:
        """Return action examples."""
        if abstract:
            if self.multiaction:
                return """\
One or several actions, separated by new lines."""
            else:
                return """\
One single action to be executed. You can only use one action at a time."""
        else:
            picked_examples = []

            # Prefer common actions.
            for action_name in ["fill", "click", "mouse_click", "keyboard_type"]:
                if action_name in self.action_set:
                    picked_examples.extend(self.action_set[action_name].examples)

            # Fallback examples.
            if not picked_examples:
                for _, action in self.action_set.items():
                    picked_examples += action.examples

            # Stable shuffle.
            rng = random.Random(1)
            rng.shuffle(picked_examples)

            if self.multiaction:
                return "\n".join(picked_examples[:max_examples])
            else:
                return picked_examples[0]

    @staticmethod
    def get_action_doc(action: HighLevelAction, with_long_description: bool=True, with_examples=True) -> str:
        """Return action prompt text."""
        if with_long_description and with_examples and action.add_code:
            description = f"""\
{action.function}
"""
        else:  
            description = f"""\
{action.signature}
"""
            if with_long_description:
                description += f"""\
    Description: {action.description}
"""
            if with_examples and action.examples:
                description += f"""\
    Examples:
"""
                for example in action.examples:
                    description += f"""\
        {example}

"""
        return description

    @staticmethod
    def get_dynamic_skill_doc(action: HighLevelAction, with_long_description: bool=True, with_examples=True) -> str:
        """Return SGDR skill prompt text."""
        if with_long_description and with_examples and action.add_code:
            description = ""
            if action.description:
                description += f"""\
{action.signature}
    Description: {action.description}
"""
            description += f"""\
{action.function}
"""
            return description
        return CustomActionSet.get_action_doc(action, with_long_description, with_examples)

            
    def describe(self, with_long_description: bool = True, with_examples: bool = True, retrieval_query: Optional[str] = None, num_retrieve: int=0) -> str:
        """Return action-space prompt text."""
        action_set = self.action_set
        if retrieval_query and num_retrieve and self.retrievable_action_set:
            retrieved_actions = self.retrieve_actions(retrieval_query, num_retrieve)
            action_set = {**action_set, **retrieved_actions}

        dyn_names = [n for n in self._dynamic_skill_names if n in action_set]
        dyn_set = set(dyn_names)
        base_items = [(n, a) for n, a in action_set.items() if n not in dyn_set]

        description = f"""
{len(base_items)} base actions are available.

"""
        for _, action in base_items:
            description += self.get_action_doc(action, with_long_description, with_examples)

        if dyn_names:
            description += (
                f"\n## Retrieved Skills\n"
                f"The following {len(dyn_names)} high-level skills were "
                f"retrieved as candidates for your next sub-routine. If "
                f"one's intent matches what you need (e.g., walking vs. "
                f"driving) and the required arguments are visible in the "
                f"accessibility tree, prefer calling it in a single "
                f"action. Otherwise proceed with primitive actions — "
                f"either way, keep making progress toward the goal.\n\n"
            )
            for n in dyn_names:
                description += self.get_dynamic_skill_doc(action_set[n], with_long_description, with_examples)

        if self.multiaction:
            description += f"""\
Multiple actions can be provided at once. An action can consume the output of a previous action by using the output variable."""
        else:
            description += f"""\
Only a single action can be provided at once."""

        return description

    @staticmethod
    def _fix_multiline_string_args(program: str) -> str:
        """Escape multiline string arguments."""
        stripped = program.strip()
        for func in ('send_msg_to_user', 'report_infeasible_instructions'):
            prefix = func + '('
            if not stripped.startswith(prefix):
                continue
            after = stripped[len(prefix):]
            if not after:
                break
            quote = after[0]
            if quote not in ('"', "'"):
                break
            # Match closing quote.
            if stripped.endswith(quote + ')'):
                inner = stripped[len(prefix) + 1 : -(1 + 1)]
                inner_fixed = inner.replace('\n', '\\n')
                return f'{func}({quote}{inner_fixed}{quote})'
        return program

    def to_python_code(self, action):
        """Convert an action string to Python code."""
        parts = action.split("```")
        if len(parts) >= 3:
            program = parts[1]
        elif len(parts) == 2:
            program = parts[1]
        else:
            # Plain code.
            program = action

        # Fix multiline messages.
        program = self._fix_multiline_string_args(program)

        # Validate code.
        python_subset_parser.parse_string(program)
        
        python_code = ""

        # Includes.
        python_code += self.python_includes

        # Final code.
        return python_code + program
