"""JSONL skill store with embedding retrieval."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from retrieval.embedder import Embedder


def _skill_id(func_name: str, code: str) -> str:
    h = hashlib.sha1(f"{func_name}\n{code}".encode("utf-8")).hexdigest()[:12]
    return f"sk_{h}"


@dataclass
class Skill:
    description: str
    code: str
    func_name: str
    website: str = ""
    src_task_id: str = ""
    window: dict = field(default_factory=dict)  # {"l": int, "t": int}
    id: str = ""
    usage_count: int = 0
    near_dup_of: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = _skill_id(self.func_name, self.code)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Skill":
        return cls(
            description=d["description"],
            code=d["code"],
            func_name=d["func_name"],
            website=d.get("website", ""),
            src_task_id=d.get("src_task_id", ""),
            window=d.get("window", {}),
            id=d.get("id", ""),
            usage_count=d.get("usage_count", 0),
            near_dup_of=list(d.get("near_dup_of", [])),
        )


# Default dedup thresholds.
TAU_HIGH_DEFAULT = None
TAU_LOW_DEFAULT = None
MMR_LAMBDA_DEFAULT = 0.7


class SkillStore:
    """JSONL skill store."""

    def __init__(
        self,
        jsonl_path: str,
        embedder: Embedder | None = None,
        tau_high: float | None = TAU_HIGH_DEFAULT,
        tau_low: float | None = TAU_LOW_DEFAULT,
    ):
        self.jsonl_path = jsonl_path
        self.embedder = embedder
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.skills: list[Skill] = []
        self._desc_emb: np.ndarray | None = None  # shape (N, dim)
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        self.load()

    # Persistence.

    def load(self) -> None:
        self.skills = []
        if not os.path.exists(self.jsonl_path):
            return
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.skills.append(Skill.from_json(json.loads(line)))
        self._desc_emb = None

    def save(self) -> None:
        tmp = self.jsonl_path + ".tmp"
        with open(tmp, "w") as f:
            for s in self.skills:
                f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
        os.replace(tmp, self.jsonl_path)

    # Mutation.

    def add(self, skill: Skill) -> tuple[bool, str]:
        """Add one skill."""
        # Name collision.
        if any(s.func_name == skill.func_name for s in self.skills):
            return False, "name_collision"

        if self.embedder is not None:
            q = self.embedder.embed_one(skill.description)
            semantic_dedup_enabled = (
                self.tau_high is not None or self.tau_low is not None
            )
            if semantic_dedup_enabled and self.skills:
                D = self._ensure_desc_emb()
                sims = D @ q
                max_idx = int(np.argmax(sims))
                max_sim = float(sims[max_idx])
                nearest = self.skills[max_idx]
                if self.tau_high is not None and max_sim >= self.tau_high:
                    return False, f"near_dup({max_sim:.3f},{nearest.func_name})"
                if (
                    self.tau_low is not None
                    and max_sim >= self.tau_low
                    and skill.src_task_id
                    and nearest.src_task_id == skill.src_task_id
                ):
                    return False, f"same_task_dup({max_sim:.3f},{nearest.func_name})"
                if self.tau_low is not None and max_sim >= self.tau_low:
                    skill.near_dup_of.append(nearest.id)
            # Update cache.
            if self._desc_emb is None or self._desc_emb.shape[0] != len(self.skills):
                self._desc_emb = np.vstack([self._ensure_desc_emb(), q[None, :]])
            else:
                self._desc_emb = np.vstack([self._desc_emb, q[None, :]])
        else:
            self._desc_emb = None

        self.skills.append(skill)
        return True, "added"

    def add_many(self, skills: Iterable[Skill]) -> dict:
        """Add multiple skills."""
        stats = {"added": 0, "skipped_name": 0, "skipped_dup": 0,
                 "skipped_same_task": 0, "near_dup_kept": 0}
        for s in skills:
            ok, reason = self.add(s)
            if ok:
                stats["added"] += 1
                if s.near_dup_of:
                    stats["near_dup_kept"] += 1
            elif reason == "name_collision":
                stats["skipped_name"] += 1
            elif reason.startswith("same_task_dup"):
                stats["skipped_same_task"] += 1
            else:
                stats["skipped_dup"] += 1
        return stats

    # Retrieval.

    def _ensure_desc_emb(self) -> np.ndarray:
        if self._desc_emb is not None and self._desc_emb.shape[0] == len(self.skills):
            return self._desc_emb
        if self.embedder is None:
            raise RuntimeError("SkillStore needs an Embedder to compute scores.")
        if not self.skills:
            self._desc_emb = np.zeros((0, self.embedder.dim), dtype=np.float32)
        else:
            descs = [s.description for s in self.skills]
            self._desc_emb = self.embedder.embed(descs, normalize=True)
        return self._desc_emb

    def topk(
        self,
        task_text: str,
        state_text: str,
        k: int = 5,
        alpha: float = 0.4,
        use_mmr: bool = True,
        mmr_lambda: float = MMR_LAMBDA_DEFAULT,
        top_m: int | None = None,
    ) -> list[tuple[Skill, float]]:
        """Retrieve top-k skills."""
        if not self.skills:
            return []
        if self.embedder is None:
            raise RuntimeError("SkillStore needs an Embedder for topk().")

        D = self._ensure_desc_emb()
        q_task = self.embedder.embed_one(task_text)
        q_state = self.embedder.embed_one(state_text)
        sim_task = D @ q_task
        sim_state = D @ q_state
        scores = alpha * sim_task + (1.0 - alpha) * sim_state

        candidate_idx = [int(i) for i in np.argsort(-scores)]

        # TopM pool.
        M = top_m if top_m is not None else max(3 * k, 20)
        if M > 0:
            candidate_idx = candidate_idx[:M]

        if not use_mmr or k <= 1 or len(candidate_idx) == 1:
            chosen = candidate_idx[:k]
            return [(self.skills[i], float(scores[i])) for i in chosen]

        # Greedy MMR.
        chosen: list[int] = [candidate_idx[0]]
        remaining = candidate_idx[1:]
        while remaining and len(chosen) < k:
            chosen_emb = D[chosen]
            rem_emb = D[remaining]
            diversity = (rem_emb @ chosen_emb.T).max(axis=1)
            rem_scores = scores[remaining]
            mmr_scores = mmr_lambda * rem_scores - (1.0 - mmr_lambda) * diversity
            best_pos = int(np.argmax(mmr_scores))
            chosen.append(remaining.pop(best_pos))

        return [(self.skills[i], float(scores[i])) for i in chosen]

    # Misc.

    def __len__(self) -> int:
        return len(self.skills)

    def func_names(self) -> set[str]:
        return {s.func_name for s in self.skills}


# Smoke test.

if __name__ == "__main__":
    import tempfile

    embedder = Embedder()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "skills.jsonl")
        store = SkillStore(path, embedder)

        ok, _ = store.add(Skill(
            description="Search for a target product on a shopping site using the search bar.",
            code="def search_product(query):\n    click('search')\n    fill('search', query)\n    keyboard_press('Enter')\n",
            func_name="search_product", website="shopping",
        ))
        assert ok
        ok, _ = store.add(Skill(
            description="Add the currently displayed product to the shopping cart.",
            code="def add_to_cart():\n    click('add-to-cart')\n",
            func_name="add_to_cart", website="shopping",
        ))
        assert ok
        ok, _ = store.add(Skill(
            description="Submit a new comment on a Reddit-like discussion post.",
            code="def post_comment(text):\n    click('reply')\n    fill('comment', text)\n    click('submit')\n",
            func_name="post_comment", website="reddit",
        ))
        assert ok

        # Name collision.
        ok, reason = store.add(Skill(
            description="Totally different description text here.",
            code="def search_product(q):\n    click('x')\n    fill('y', q)\n",
            func_name="search_product", website="shopping",
        ))
        assert not ok and reason == "name_collision", (ok, reason)

        # Paraphrase insert.
        ok, reason = store.add(Skill(
            description="Use the search bar of a shopping site to look up a target product.",
            code="def find_product(query):\n    click('q')\n    fill('q', query)\n    keyboard_press('Enter')\n",
            func_name="find_product", website="shopping",
        ))
        assert ok and reason == "added", (ok, reason)
        store.save()

        # Reload and query.
        store2 = SkillStore(path, embedder)

        results = store2.topk(
            task_text="Find a black running shoe and add it to my cart.",
            state_text="Page is a shopping homepage with a search bar and product listings.",
            k=3, alpha=0.4, use_mmr=True,
        )
        print(f"\nTop-{len(results)} (MMR on):")
        for sk, score in results:
            print(f"  [{score:.3f}] {sk.func_name}: {sk.description}")

        # Sanity check.
        top_names = [sk.func_name for sk, _ in results]
        assert top_names[0] in {"search_product", "add_to_cart"}, top_names
        assert top_names[-1] == "post_comment", top_names
        print("\nSmoke test OK.")
        embedder.flush()
