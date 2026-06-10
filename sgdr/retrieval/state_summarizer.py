"""State summarizer for SGDR retrieval."""

from __future__ import annotations

import hashlib
import logging
import re
import threading

from llm_client import llm_completion

logger = logging.getLogger(__name__)


_SYS_PROMPT = (
    "You are a state summarizer for a web agent whose action library is "
    "indexed by descriptions like 'submit a forum post on a submission "
    "form' or 'apply a price filter on a product listing page'. Your "
    "summary will be cosine-matched against such skill descriptions, so "
    "use the SAME operational vocabulary they do.\n\n"
    "Given the current page's accessibility tree (axtree) plus the URL "
    "and title, produce ONE short paragraph (1-2 sentences) that:\n"
    "1. Names the kind of page in operational terms (e.g. 'forum "
    "submission form', 'product listing page', 'opened forum-selection "
    "combobox', 'post-detail page with comment section').\n"
    "2. Lists the action verbs this page ENABLES right now — i.e. what "
    "sub-routines could plausibly run on this exact state. Use verb + "
    "object phrasing (e.g. 'submit a post', 'select a forum', 'fill in "
    "the title and body', 'open the sort menu', 'apply a filter').\n\n"
    "Do NOT enumerate every visible element, do NOT describe pure visuals "
    "(colors, layout), and do NOT mention task instructions or speculate "
    "about future steps. Output only the summary text."
)


# Normalize AXTree ids.
_AXTREE_ID_RE = re.compile(r"\[\d+\]\s*")


def _axtree_outline(axtree_txt: str) -> str:
    return _AXTREE_ID_RE.sub("", axtree_txt)


def _hash_state(axtree_txt: str, url: str = "") -> str:
    return hashlib.sha256(
        f"{url}\n{_axtree_outline(axtree_txt)}".encode("utf-8")
    ).hexdigest()


class StateSummarizer:
    """LLM-backed state summarizer."""

    def __init__(
        self,
        model: str,
        max_axtree_chars: int = 12000,
        max_tokens: int = 160,
        temperature: float = 0.0,
    ):
        self.model = model
        self.max_axtree_chars = max_axtree_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    # Public API.

    def summarize(self, obs: dict) -> str:
        """Summarize one observation."""
        axtree_txt = obs.get("axtree_txt", "") or ""
        urls = obs.get("open_pages_urls", []) or []
        titles = obs.get("open_pages_titles", []) or []
        active = obs.get("active_page_index", 0) or 0
        url = urls[active] if 0 <= active < len(urls) else ""
        title = titles[active] if 0 <= active < len(titles) else ""

        key = _hash_state(axtree_txt, url)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        summary = self._call_llm(url, title, axtree_txt)
        if not summary:
            summary = self._url_title_stub(url, title)

        with self._lock:
            self._cache[key] = summary
        return summary

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    # Internals.

    def _call_llm(self, url: str, title: str, axtree_txt: str) -> str:
        truncated = axtree_txt
        if len(truncated) > self.max_axtree_chars:
            truncated = truncated[: self.max_axtree_chars] + "\n... [truncated]"

        user_msg = (
            f"URL: {url}\n"
            f"Page title: {title}\n\n"
            f"Accessibility tree:\n{truncated}"
        )
        try:
            resp = llm_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYS_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("[StateSummarizer] LLM call failed: %s", e)
            return ""

    def _url_title_stub(self, url: str, title: str) -> str:
        if title and url:
            return f"Page titled '{title}' at {url}."
        if url:
            return f"Web page at {url}."
        return "An unknown web page."


# ── smoke test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _self = sys.modules[__name__]  # the live module — required because
                                   # `python -m` makes a fresh copy if we
                                   # `import induce.state_summarizer`.

    # 1. Hash determinism
    h1 = _hash_state("axtree-A", "http://x")
    h2 = _hash_state("axtree-A", "http://x")
    h3 = _hash_state("axtree-B", "http://x")
    assert h1 == h2 != h3, (h1, h2, h3)

    # 2. URL/title stub when LLM raises
    summ = StateSummarizer(model="dummy")
    obs = {
        "axtree_txt": "[1] button 'Submit'\n[2] textbox 'Search'",
        "open_pages_urls": ["http://shop.example/laptops"],
        "open_pages_titles": ["Laptops - Shop"],
        "active_page_index": 0,
    }
    out_stub = summ.summarize(obs)
    print(f"[stub] -> {out_stub!r}")
    assert "shop.example/laptops" in out_stub
    assert summ.cache_size() == 1

    # 3. Stub the LLM and verify cache hits avoid further calls
    n_calls = {"count": 0}

    class _Choice:
        def __init__(self, text):
            self.message = type("M", (), {"content": text})()

    class _Resp:
        def __init__(self, text):
            self.choices = [_Choice(text)]

    def fake_completion(**kwargs):
        n_calls["count"] += 1
        return _Resp("Shopping product listing page with a search bar and Submit button.")

    _self.llm_completion = fake_completion  # monkeypatch

    summ2 = StateSummarizer(model="dummy")
    a = summ2.summarize(obs)
    b = summ2.summarize(obs)  # cache hit
    print(f"[stub]     -> {a!r}")
    assert a == b
    assert "search bar" in a.lower()
    assert n_calls["count"] == 1, f"expected 1 LLM call, got {n_calls['count']}"

    # 4. Different obs → new LLM call
    obs2 = dict(obs, axtree_txt="[3] link 'Home'")
    c = summ2.summarize(obs2)
    assert n_calls["count"] == 2, f"expected 2 LLM calls, got {n_calls['count']}"

    print("Smoke test OK.")
