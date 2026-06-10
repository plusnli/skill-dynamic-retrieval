"""Generate resolved test configs."""

import json
import os

BASE_URL = os.environ.get("BASE_URL", "http://localhost")


def site_url(env_name: str, fallback: str) -> str:
    """Return a site URL."""
    return os.environ.get(env_name, fallback).rstrip("/")


SHOPPING = site_url("WA_SHOPPING", f"{BASE_URL}:8082")
SHOPPING_ADMIN = site_url("WA_SHOPPING_ADMIN", f"{BASE_URL}:8083/admin")
REDDIT = site_url("WA_REDDIT", f"{BASE_URL}:8080")
GITLAB = site_url("WA_GITLAB", f"{BASE_URL}:9001")
MAP = site_url("WA_MAP", f"{BASE_URL}:443")
WIKIPEDIA = site_url(
    "WA_WIKIPEDIA",
    f"{BASE_URL}:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
)


def main() -> None:
    with open("config_files/test.raw.json", "r") as f:
        raw = f.read()
    raw = raw.replace("__GITLAB__", GITLAB)
    raw = raw.replace("__REDDIT__", REDDIT)
    raw = raw.replace("__SHOPPING__", SHOPPING)
    raw = raw.replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
    raw = raw.replace("__WIKIPEDIA__", WIKIPEDIA)
    raw = raw.replace("__MAP__", MAP)
    with open("config_files/test.json", "w") as f:
        f.write(raw)
    # Per-task files.
    data = json.loads(raw)
    for idx, item in enumerate(data):
        with open(f"config_files/{idx}.json", "w") as f:
            json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()
