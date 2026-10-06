import json
import re
from pathlib import Path
from typing import Any

import yaml


SECRET_KEY_FRAGMENTS = ("api_key", "access_token", "secret", "password", "token")
REDACTED = "<redacted>"

_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9._-]+"),
    re.compile(r"\bsecret-[A-Za-z0-9._-]+"),
    re.compile(r"\bxox[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(api_key\s*=\s*)[^\s,;]+"),
)


def load_local_config(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Local config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            loaded = json.load(handle)
        else:
            loaded = yaml.safe_load(handle) or {}

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Local config must be a mapping: {config_path}")
    return loaded


def _is_secret_key(key: Any) -> bool:
    key_text = str(key).lower()
    return any(fragment in key_text for fragment in SECRET_KEY_FRAGMENTS)


def redact_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, child in value.items():
            redacted[key] = REDACTED if _is_secret_key(key) else redact_mapping(child)
        return redacted
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_mapping(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(text: str) -> str:
    redacted = text
    for pattern in _CREDENTIAL_PATTERNS:
        if "api_key" in pattern.pattern:
            redacted = pattern.sub(r"\1" + REDACTED, redacted)
        else:
            redacted = pattern.sub(REDACTED, redacted)
    return redacted
