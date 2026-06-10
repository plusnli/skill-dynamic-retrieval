from __future__ import annotations

import re

from llm_client import llm_completion
from retrieval.cer_store import CERStore, DynamicsExperience, SkillExperience


_DYNAMICS_RETRIEVER_SYS = """You are a retrieval module for a web agent.
Pick the most useful environment dynamics for the current task.
Return only IDs that are directly helpful for this goal and website context."""

_SKILLS_RETRIEVER_SYS = """You are a retrieval module for a web agent.
Pick the most useful reusable skills for the current task.
Return only IDs that are directly helpful for this goal and website context."""


def _parse_selected_ids(text: str) -> list[int]:
    ids = []
    for x in re.findall(r"\b\d+\b", text or ""):
        ids.append(int(x))
    out = []
    seen = set()
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


class CERRetriever:
    def __init__(self, model: str, top_k_dynamics: int = 5, top_k_skills: int = 5):
        self.model = model
        self.top_k_dynamics = top_k_dynamics
        self.top_k_skills = top_k_skills

    def _retrieve_dynamics(
        self,
        goal: str,
        website_desc: str,
        dynamics: list[DynamicsExperience],
    ) -> list[DynamicsExperience]:
        if not dynamics:
            return []
        candidates = []
        for idx, d in enumerate(dynamics):
            candidates.append(
                f"{idx}: {d.name}\nDescription: {d.description}\nUsages: {d.usages}\nURL: {d.url}"
            )
        user_msg = (
            f"Task goal:\n{goal}\n\nWebsite description:\n{website_desc}\n\n"
            f"Dynamics list:\n" + "\n\n".join(candidates) + "\n\n"
            f"Select at most {self.top_k_dynamics} ids.\n"
            "Respond in this format:\n<selected-pages>id1;id2;id3</selected-pages>"
        )
        resp = llm_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": _DYNAMICS_RETRIEVER_SYS},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        ids = _parse_selected_ids(text)
        out = []
        for i in ids:
            if 0 <= i < len(dynamics):
                out.append(dynamics[i])
            if len(out) >= self.top_k_dynamics:
                break
        return out

    def _retrieve_skills(
        self,
        goal: str,
        website_desc: str,
        skills: list[SkillExperience],
    ) -> list[SkillExperience]:
        if not skills:
            return []
        candidates = []
        for idx, s in enumerate(skills):
            steps_preview = "\n".join(f"- {x}" for x in s.steps[:5])
            candidates.append(f"{idx}: {s.name}\nSteps:\n{steps_preview}")
        user_msg = (
            f"Task goal:\n{goal}\n\nWebsite description:\n{website_desc}\n\n"
            f"Skills list:\n" + "\n\n".join(candidates) + "\n\n"
            f"Select at most {self.top_k_skills} ids.\n"
            "Respond in this format:\n<selected-skills>id1;id2;id3</selected-skills>"
        )
        resp = llm_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": _SKILLS_RETRIEVER_SYS},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        ids = _parse_selected_ids(text)
        out = []
        for i in ids:
            if 0 <= i < len(skills):
                out.append(skills[i])
            if len(out) >= self.top_k_skills:
                break
        return out

    def retrieve(self, goal: str, website_desc: str, store: CERStore) -> tuple[list[DynamicsExperience], list[SkillExperience]]:
        dynamics = self._retrieve_dynamics(goal, website_desc, store.dynamics)
        skills = self._retrieve_skills(goal, website_desc, store.skills)
        return dynamics, skills

