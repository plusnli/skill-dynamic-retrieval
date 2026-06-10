import re
import time

import playwright.sync_api

from browsergym.core.env import BrowserEnv, logger


agent_args = None

def execute_python_code(
    code: str,
    page: playwright.sync_api.Page,
    send_message_to_user: callable,
    report_infeasible_instructions: callable,
    **additional_globals
):
    """Execute action code with browser globals."""

    globals = {
        "page": page,
        "send_message_to_user": send_message_to_user,
        "report_infeasible_instructions": report_infeasible_instructions,
        **additional_globals,
    }

    exec(code, globals)


def step(self: BrowserEnv, action: str) -> tuple:
    """BrowserEnv step with custom exec."""
    self.last_action = action

    info = {}
    info["action_exec_start"] = time.time()
    info["action_exec_timeout"] = 0

    def send_message_to_user(text: str):
        if not isinstance(text, str):
            raise ValueError(f"Forbidden value: {text} is not a string")
        self.chat.add_message(role="assistant", msg=text)

    def report_infeasible_instructions(reason: str):
        if not isinstance(reason, str):
            raise ValueError(f"Forbidden value: {reason} is not a string")
        self.chat.add_message(role="infeasible", msg=reason)
        self.infeasible_message_received = True

    # Execute action.
    logger.debug(f"Executing action")
    try:
        if self.action_mapping:
            code = self.action_mapping(action)
        else:
            code = action
        execute_python_code(
            code,
            self.page,
            send_message_to_user=send_message_to_user,
            report_infeasible_instructions=report_infeasible_instructions,
            agent_args=agent_args,
            env=self,
        )
        self.last_action_error = ""
    except Exception as e:
        self.last_action_error = f"{type(e).__name__}: {e}"
        match = re.match("TimeoutError: Timeout ([0-9]+)ms exceeded.", self.last_action_error)
        if match:
            info["action_exec_timeout"] = float(match.groups()[0]) / 1000
    logger.debug(f"Action executed")
    info["action_exec_stop"] = time.time()

    # Let JS callbacks settle.
    time.sleep(0.5)
    self.context.cookies()

    # Wait for DOM.
    self._wait_dom_loaded()

    # Check active page.
    self._active_page_check()
    logger.debug(f"Active page checked")

    # Wait for user.
    self._wait_for_user_message()
    logger.debug(f"User message done")

    logger.debug(f"Initiating task validation")
    # Validate task.
    reward, done, user_message, task_info = self._task_validate()
    info["task_info"] = task_info
    logger.debug(f"Task validation done")

    # Add task message.
    if user_message:
        self.chat.add_message(role="user", msg=user_message)

    # Get observation.
    obs = self._get_obs()
    logger.debug(f"Observation extracted")

    # Gymnasium API.
    terminated = done or (
        self.terminate_on_infeasible and self.infeasible_message_received
    )
    truncated = False

    return obs, reward, terminated, truncated, info
    
def patch_with_custom_exec(args):
    global agent_args
    agent_args = args
    setattr(BrowserEnv, "step", step)
