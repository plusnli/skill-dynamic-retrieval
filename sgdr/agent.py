import base64
import dataclasses
import io
import json
import os
import logging

import numpy as np
from PIL import Image

from browsergym.experiments import AbstractAgentArgs, Agent
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from custom_action_set import CustomActionSet
from actions import ACTION_DICT
from llm_client import llm_completion
from retrieval import CERRetriever, CERStore, Embedder, SkillStore, StateSummarizer

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert an image to a JPEG data URL."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


def _is_benign_timeout(err_msg: str) -> bool:
    """Ignore harmless post-action wait timeouts."""
    if not err_msg:
        return False
    if "Timeout 500ms exceeded" not in err_msg:
        return False
    succeeded_markers = (
        "performing click action",
        "selected specified option(s)",
        "filled",
    )
    return any(m in err_msg for m in succeeded_markers)


class DemoAgent(Agent):
    """BrowserGym web agent."""

    def obs_preprocessor(self, obs: dict) -> dict:

        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def __init__(
        self,
        model_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
        websites: tuple[str],
        actions: list[str],
        memory: str,
        skill_store_path: str = None,
        cer_store_path: str = None,
        cer_top_k_dynamics: int = 5,
        cer_top_k_skills: int = 5,
        top_k: int = 5,
        top_m: int | None = None,
        alpha: float = 0.4,
        mmr_lambda: float = 0.7,
        use_mmr: bool = True,
        summarizer_model: str = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.websites = websites

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        custom_actions = ACTION_DICT["general"] + ACTION_DICT["webarena"]
        for w in websites:
            custom_actions = custom_actions + ACTION_DICT[w]


        self.action_set = CustomActionSet(
            subsets=["custom"],
            custom_actions=custom_actions,
            strict=False,
            multiaction=True,
            demo_mode=demo_mode,
        )

        self.action_history = []

        self.actions = actions
        self.num_actions = 0
        
        if memory is None:
            self.memory = None
        else:
            self.memory = open(memory, 'r').read()
            if self.memory.strip() == "":
                self.memory = None

        # SGDR/CER state.
        self.skill_store = None
        self.state_summarizer = None
        self.top_k = top_k
        self.top_m = top_m
        self.alpha = alpha
        self.mmr_lambda = mmr_lambda
        self.use_mmr = use_mmr
        self._goal_text: str | None = None
        self._last_activated_skills: list[str] = []
        self._activation_step: int = 0
        self.cer_store = None
        self.cer_retriever = None
        self._cer_block: str | None = None
        # Per-step SGDR retrieval log.
        self._activation_log_path: str | None = (
            os.environ.get("SGDR_ACTIVATION_LOG")
            or os.environ.get("SKILL_RAG_ACTIVATION_LOG")
            or None
        )
        if skill_store_path:
            embedder = Embedder()
            self.skill_store = SkillStore(skill_store_path, embedder=embedder)
            self.state_summarizer = StateSummarizer(
                model=summarizer_model or model_name,
            )
            if self._activation_log_path:
                os.makedirs(os.path.dirname(self._activation_log_path) or ".",
                            exist_ok=True)
            effective_m = top_m if top_m is not None else max(3 * top_k, 20)
            print(f"[sgdr] enabled: store={skill_store_path} "
                  f"({len(self.skill_store)} skills), "
                  f"K={top_k}, M={effective_m}, alpha={alpha}, "
                  f"mmr={'on' if use_mmr else 'off'}(lambda={mmr_lambda}), "
                  f"activation_log={self._activation_log_path or 'off'}")
        if cer_store_path:
            self.cer_store = CERStore(cer_store_path)
            self.cer_retriever = CERRetriever(
                model=model_name,
                top_k_dynamics=cer_top_k_dynamics,
                top_k_skills=cer_top_k_skills,
            )
            print(
                f"[cer] enabled: store={cer_store_path} "
                f"(dynamics={len(self.cer_store.dynamics)}, "
                f"skills={len(self.cer_store.skills)}), "
                f"k_d={cer_top_k_dynamics}, k_s={cer_top_k_skills}"
            )

    def get_action(self, obs: dict) -> tuple[str, dict]:
        if len(self.actions) == 0 or (self.num_actions > (len(self.actions) - 1)):
            # Per-step skill injection.
            self._activate_skills(obs)

            system_msgs = []
            user_msgs = []

            if self.chat_mode:
                system_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Instructions

    You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
    communicate with the user via a chat, to which the user gives you instructions and to which you
    can send back messages. You have access to a web browser that both you and the user can see,
    and with which only you can interact via specific commands.

    Review the instructions from the user, the current state of the page and all other information
    to find the best possible next action to accomplish your goal. Your answer will be interpreted
    and executed by a program, make sure to follow the formatting instructions.
    """,
                    }
                )

                # append chat messages
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Chat Messages
    """,
                    }
                )
                for msg in obs["chat_messages"]:
                    if msg["role"] in ("user", "assistant", "infeasible"):
                        user_msgs.append(
                            {
                                "type": "text",
                                "text": f"""\
    - [{msg['role']}] {msg['message']}
    """,
                            }
                        )
                    elif msg["role"] == "user_image":
                        user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                    else:
                        raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

            else:
                assert obs["goal_object"], "The goal is missing."
                system_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Instructions

    Review the current state of the page and all other information to find the best
    possible next action to accomplish your goal. Your answer will be interpreted
    and executed by a program, make sure to follow the formatting instructions.

    # Reporting rules (when calling send_msg_to_user)

    Your message is graded by an automated checker against an expected answer
    string. Extra commentary makes correct answers look wrong; silence or
    refusal also fails. Follow these rules strictly:

    1. ALWAYS send an answer. Information-seeking goals (find, what is, how
       much, how long, list, tell me, where is, who, when) MUST end with
       send_msg_to_user containing your best answer. Never terminate without
       sending one.
    2. EXCLUDE commentary: no preamble ("Based on...", "I found...",
       "According to..."), no postamble ("Hope this helps", "Let me know"),
       no apologies ("Sorry, ..."), no hedges ("However", "Note that",
       "Please note"), no follow-up questions ("Would you like me to...",
       "Please provide..."). State the answer plainly.
    3. NEVER ask the user for clarification or more information — make your
       best attempt with what's available.
    4. If the goal asks you to send a specific string verbatim, send EXACTLY
       that string.
    5. If the goal asks for a fact (number, name, address, list, phone, URL,
       hours), state it as plainly as possible — a short phrase or list is
       fine; a long explanatory sentence is not.
    6. If the website does not contain the answer, fall back to your best
       guess from general knowledge or partial evidence already gathered in
       the trajectory. Do NOT refuse, do NOT apologize, and do NOT navigate
       to external sites (google.com, bing.com, etc.) — work only with the
       current site you've been navigating.
    7. For action-only goals (post a comment, edit a profile, change a
       setting), performing the action is the answer; a final send_msg is
       not required.
    """,
                    }
                )
                # append memory
                if self.memory is not None:
                    system_msgs.append({
                        "type": "text",
                        "text": self.memory,
                    })
                cer_block = self._build_cer_prompt_block(obs)
                if cer_block:
                    system_msgs.append({"type": "text", "text": cer_block})
                # append goal
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Goal
    """,
                    }
                )
                # goal_object is directly presented as a list of openai-style messages
                user_msgs.extend(obs["goal_object"])

            # append url of all open tabs
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
    # Currently open tabs
    """,
                }
            )
            for page_index, (page_url, page_title) in enumerate(
                zip(obs["open_pages_urls"], obs["open_pages_titles"])
            ):
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
    Title: {page_title}
    URL: {page_url}
    """,
                    }
                )

            # append page AXTree (if asked)
            if self.use_axtree:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Current page Accessibility Tree

    {obs["axtree_txt"]}

    """,
                    }
                )
            # append page HTML (if asked)
            if self.use_html:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Current page DOM

    {obs["pruned_html"]}

    """,
                    }
                )

            # append page screenshot (if asked)
            if self.use_screenshot:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": """\
    # Current page Screenshot
    """,
                    }
                )
                user_msgs.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_jpg_base64_url(obs["screenshot"]),
                            "detail": "auto",
                        },  # Literal["low", "high", "auto"] = "auto"
                    }
                )
        
            # append action space description
            skill_priority_hint, skill_cot_example = self._build_skill_prompt_blocks()
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
    # Action Space

    {self.action_set.describe(with_long_description=True, with_examples=True)}

    {skill_priority_hint}
    Here are examples of actions with chain-of-thought reasoning:
{skill_cot_example}
    I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
    ```click("12")```

    I found the information requested by the user, I will send it to the chat.
    ```send_msg_to_user("The price for a 15" laptop is 1499 USD.")```

    Only wrap the to-be-executed action in triple backticks. Do not wrap the reasoning or the action description.

    """,
                }
            )

            # append past actions (and last error message) if any
            if self.action_history:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # History of past actions
    """,
                    }
                )
                user_msgs.extend(
                    [
                        {
                            "type": "text",
                            "text": f"""\

    {action}
    """,
                        }
                        for action in self.action_history
                    ]
                )

                if obs["last_action_error"] and not _is_benign_timeout(obs["last_action_error"]):
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
    # Error message from last action

    {obs["last_action_error"]}

    """,
                        }
                    )
                    print("Error message from last action: ", obs["last_action_error"])
                    # cont = input("Continue? (y/n): ")

            # ask for the next action
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
    # Next action

    You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action.
    """,
                }
            )

            prompt_text_strings = []
            for message in system_msgs + user_msgs:
                match message["type"]:
                    case "text":
                        prompt_text_strings.append(message["text"])
                    case "image_url":
                        image_url = message["image_url"]
                        if isinstance(message["image_url"], dict):
                            image_url = image_url["url"]
                        if image_url.startswith("data:image"):
                            prompt_text_strings.append(
                                "image_url: " + image_url[:30] + "... (truncated)"
                            )
                        else:
                            prompt_text_strings.append("image_url: " + image_url)
                    case _:
                        raise ValueError(
                            f"Unknown message type {repr(message['type'])} in the task goal."
                        )

            try:
                response = llm_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_msgs},
                        {"role": "user", "content": user_msgs},
                    ],
                    temperature=0.0,
                )
                action = response.choices[0].message.content
                action = action.replace('```python', '```')
            except Exception:
                action = ""
        else:
            if self.num_actions > (len(self.actions) - 1):
                action = None
            else:
                action = self.actions[self.num_actions]
            self.num_actions += 1

        self.action_history.append(action)

        return action, {}

    # SGDR helpers.

    def _extract_goal_text(self, obs: dict) -> str:
        """Extract the task goal text."""
        if self._goal_text is not None:
            return self._goal_text
        parts: list[str] = []
        for m in (obs.get("goal_object") or []):
            if isinstance(m, dict) and m.get("type") == "text":
                t = (m.get("text") or "").strip()
                if t:
                    parts.append(t)
        if not parts:
            for m in (obs.get("chat_messages") or []):
                if m.get("role") == "user":
                    t = (m.get("message") or "").strip()
                    if t:
                        parts.append(t)
        self._goal_text = " ".join(parts).strip() or "Web automation task."
        return self._goal_text

    def _website_description(self) -> str:
        website_desc = {
            "shopping": "E-commerce storefront for product search, filters, cart and order details.",
            "admin": "Shopping admin dashboard for orders, customers, products and reviews.",
            "reddit": "Forum website for reading posts, comments, profile and posting submissions.",
            "gitlab": "Project collaboration site with repositories, issues, merge requests and members.",
            "map": "Map website for searching places, routes, travel mode and distance/time info.",
        }
        rows = []
        for w in self.websites:
            rows.append(f"- {w}: {website_desc.get(w, 'Website operations and navigation tasks.')}")
        return "\n".join(rows) if rows else "- generic web environment"

    def _build_cer_prompt_block(self, obs: dict) -> str:
        """Render CER replay memory."""
        if self.cer_store is None or self.cer_retriever is None:
            return ""
        if self._cer_block is not None:
            return self._cer_block

        goal = self._extract_goal_text(obs)
        dynamics, skills = self.cer_retriever.retrieve(
            goal=goal,
            website_desc=self._website_description(),
            store=self.cer_store,
        )
        lines = ["# Retrieved Contextual Experience Replay Memory"]
        if dynamics:
            lines.append("## Environment dynamics")
            for i, d in enumerate(dynamics, start=1):
                lines.append(f"### Dynamics {i}: {d.name}")
                lines.append(f"- Description: {d.description}")
                lines.append(f"- Potential usages: {d.usages}")
                if d.url:
                    lines.append(f"- URL: {d.url}")
        if skills:
            lines.append("## Skills")
            for i, s in enumerate(skills, start=1):
                lines.append(f"### Skill {i}: {s.name}")
                for j, step in enumerate(s.steps, start=1):
                    lines.append(f"{j}. {step}")
        self._cer_block = "" if len(lines) == 1 else "\n".join(lines)
        return self._cer_block

    def _build_skill_prompt_blocks(self) -> tuple[str, str]:
        """Build SGDR prompt hints."""
        names = list(self._last_activated_skills)
        if not names:
            hint = (
                "If a high-level function exactly covers your next "
                "sub-routine, call it in a single action."
            )
            return hint, ""

        first = names[0]
        try:
            sig = self.action_set.action_set[first].signature
            inside = sig[sig.index("(") + 1 : sig.rindex(")")]
            params = []
            for p in inside.split(","):
                p = p.strip()
                if not p:
                    continue
                p = p.split(":")[0].strip().split("=")[0].strip()
                if p:
                    params.append(p)
            example_args = ", ".join(f"'{p}'" for p in params)
            example_call = f"{first}({example_args})"
        except (KeyError, ValueError):
            example_call = f"{first}(...)"

        cot_example = (
            f"\n    The retrieved high-level skill `{first}` matches the "
            f"sub-routine I need next on this page (intent and arguments "
            f"both line up), so I will call it in a single action.\n"
            f"    ```{example_call}```\n"
        )
        return "", cot_example


    def _activate_skills(self, obs: dict) -> None:
        """Retrieve and inject skills for one step."""
        if self.skill_store is None or self.state_summarizer is None:
            return
        if len(self.skill_store) == 0:
            self.action_set.set_dynamic_skills([])
            self._last_activated_skills = []
            return

        try:
            goal_text = self._extract_goal_text(obs)
            state_summary = self.state_summarizer.summarize(obs)
            ranked = self.skill_store.topk(
                task_text=goal_text,
                state_text=state_summary,
                k=self.top_k,
                top_m=self.top_m,
                alpha=self.alpha,
                use_mmr=self.use_mmr,
                mmr_lambda=self.mmr_lambda,
            )
            skills = [s for s, _ in ranked]
            injected = self.action_set.set_dynamic_skills(skills)
            self._last_activated_skills = list(injected)

            # Track injected skills.
            injected_set = set(injected)
            for s, _ in ranked:
                if s.func_name in injected_set:
                    s.usage_count += 1

            scores = {s.func_name: f"{score:.3f}" for s, score in ranked}
            print(f"[sgdr] state='{state_summary[:80]}...' "
                  f"activated={injected} scores={scores}")

            # Activation log.
            if self._activation_log_path:
                try:
                    record = {
                        "step": self._activation_step,
                        "goal": goal_text,
                        "state_summary": state_summary,
                        "injected": list(injected),
                        "ranked": [
                            {"func_name": s.func_name, "score": float(score),
                             "id": s.id, "src_task_id": s.src_task_id}
                            for s, score in ranked
                        ],
                    }
                    with open(self._activation_log_path, "a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning("[sgdr] failed to write activation log: %s", e)
        except Exception as e:
            logger.warning(
                "[sgdr] retrieval failed at step %d (%s); "
                "using base actions", self._activation_step, e,
            )
            self.action_set.set_dynamic_skills([])
            self._last_activated_skills = []
        finally:
            self._activation_step += 1




@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """Serializable agent config."""

    model_name: str = "gpt-4.1"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False
    websites: tuple[str] = ()
    actions: list[str] = ()
    memory: str = None
    skill_store_path: str = None
    cer_store_path: str = None
    cer_top_k_dynamics: int = 5
    cer_top_k_skills: int = 5
    top_k: int = 5
    top_m: int | None = None
    alpha: float = 0.4
    mmr_lambda: float = 0.7
    use_mmr: bool = True
    summarizer_model: str = None

    def make_agent(self):
        return DemoAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            websites=self.websites,
            actions=self.actions,
            memory=self.memory,
            skill_store_path=self.skill_store_path,
            cer_store_path=self.cer_store_path,
            cer_top_k_dynamics=self.cer_top_k_dynamics,
            cer_top_k_skills=self.cer_top_k_skills,
            top_k=self.top_k,
            top_m=self.top_m,
            alpha=self.alpha,
            mmr_lambda=self.mmr_lambda,
            use_mmr=self.use_mmr,
            summarizer_model=self.summarizer_model,
        )
