from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field


def _stable_id(prefix: str, content: str) -> str:
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


@dataclass
class DynamicsExperience:
    name: str
    description: str
    usages: str
    url: str
    src_task_id: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.name}\n{self.url}\n{self.description}\n{self.usages}"
            self.id = _stable_id("dyn", key)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "DynamicsExperience":
        return cls(
            name=d.get("name", "").strip(),
            description=d.get("description", "").strip(),
            usages=d.get("usages", "").strip(),
            url=d.get("url", "").strip(),
            src_task_id=d.get("src_task_id", "").strip(),
            id=d.get("id", "").strip(),
        )


@dataclass
class SkillExperience:
    name: str
    steps: list[str] = field(default_factory=list)
    src_task_id: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            key = f"{self.name}\n" + "\n".join(self.steps)
            self.id = _stable_id("sk", key)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SkillExperience":
        return cls(
            name=d.get("name", "").strip(),
            steps=[str(s).strip() for s in d.get("steps", []) if str(s).strip()],
            src_task_id=d.get("src_task_id", "").strip(),
            id=d.get("id", "").strip(),
        )


class CERStore:
    """Persistent experience buffer for CER online replay."""

    def __init__(self, json_path: str):
        self.json_path = json_path
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        self.dynamics: list[DynamicsExperience] = []
        self.skills: list[SkillExperience] = []
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.json_path):
            self.dynamics = []
            self.skills = []
            return
        raw = json.load(open(self.json_path))
        self.dynamics = [DynamicsExperience.from_json(x) for x in raw.get("dynamics", [])]
        self.skills = [SkillExperience.from_json(x) for x in raw.get("skills", [])]

    def save(self) -> None:
        payload = {
            "dynamics": [x.to_json() for x in self.dynamics],
            "skills": [x.to_json() for x in self.skills],
        }
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.json_path)

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    def add_dynamics(self, items: list[DynamicsExperience]) -> dict:
        existing_urls = {self._norm(x.url) for x in self.dynamics if x.url}
        existing_desc = {self._norm(f"{x.name} {x.description}") for x in self.dynamics}
        stats = {"added": 0, "skipped": 0}
        for it in items:
            if not it.name or not it.description:
                stats["skipped"] += 1
                continue
            key_desc = self._norm(f"{it.name} {it.description}")
            key_url = self._norm(it.url)
            if (key_url and key_url in existing_urls) or key_desc in existing_desc:
                stats["skipped"] += 1
                continue
            self.dynamics.append(it)
            existing_desc.add(key_desc)
            if key_url:
                existing_urls.add(key_url)
            stats["added"] += 1
        return stats

    def add_skills(self, items: list[SkillExperience]) -> dict:
        existing_names = {self._norm(x.name) for x in self.skills}
        existing_steps = {self._norm("\n".join(x.steps)) for x in self.skills}
        stats = {"added": 0, "skipped": 0}
        for it in items:
            if not it.name or not it.steps:
                stats["skipped"] += 1
                continue
            key_name = self._norm(it.name)
            key_steps = self._norm("\n".join(it.steps))
            if key_name in existing_names or key_steps in existing_steps:
                stats["skipped"] += 1
                continue
            self.skills.append(it)
            existing_names.add(key_name)
            existing_steps.add(key_steps)
            stats["added"] += 1
        return stats

