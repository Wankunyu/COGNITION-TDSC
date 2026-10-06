"""One request per puzzle, explicit provider payloads, complete usage accounting."""
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from .tasks import ROOT, prompt_for


class ProviderError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def api_key(provider, secrets_path=None):
    name = "GEMINI_API_KEY" if provider == "gemini" else "CAPTCHAMIND_API_KEY"
    key = os.environ.get(name, "")
    if not key and provider == "gemini":
        key = os.environ.get("GOOGLE_API_KEY", "")
    if not key and secrets_path:
        values = yaml.safe_load(Path(secrets_path).read_text()) or {}
        key = values.get("providers", {}).get(provider, {}).get("api_key", "")
    return key


def request_json(url, payload, headers, timeout):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode(errors="replace")
        for value in headers.values():
            if value and value != "application/json":
                detail = detail.replace(value, "[redacted]")
                if value.startswith("Bearer "):
                    detail = detail.replace(value[7:], "[redacted]")
        raise ProviderError(f"HTTP {exc.code}: {detail}", exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"Transport failure: {type(exc).__name__}") from None


class Provider:
    def __init__(self, name, config, secrets_path=None, base_url=None, prompt_builder=None):
        self.name, self.config = name, config
        self.settings = config[name]
        self.model = self.settings["model"]
        self.key = api_key(name, secrets_path)
        self.timeout = config["request_timeout_seconds"]
        self.prompt_builder = prompt_builder or prompt_for
        self.base_url = (base_url or os.environ.get(self.settings.get("base_url_env", ""), "")).rstrip("/")
        if name == "gemini" and not self.key:
            raise ProviderError("Gemini API key is not configured")
        if name == "captchamind" and not self.base_url:
            raise ProviderError("Set CAPTCHAMIND_BASE_URL to the configured inference endpoint")

    def check(self):
        if self.name == "gemini":
            return request_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}", None,
                {"x-goog-api-key": self.key}, min(self.timeout, 30),
            )
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        return request_json(self.base_url + "/models", None, headers, 30)

    def solve(self, row):
        prompt = self.prompt_builder(row)
        images = []
        for image_path in row["images"]:
            path = ROOT / image_path
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            images.append((mime, base64.b64encode(path.read_bytes()).decode()))
        headers = {"Content-Type": "application/json"}
        if self.name == "gemini":
            parts = [{"text": prompt}] + [
                {"inlineData": {"mimeType": mime, "data": data}} for mime, data in images
            ]
            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": self.settings["temperature"],
                    "maxOutputTokens": self.config["max_output_tokens"],
                    "thinkingConfig": {"thinkingLevel": self.settings["thinking_level"]},
                },
            }
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            headers["x-goog-api-key"] = self.key
        else:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
                for mime, data in images
            ]
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.settings["temperature"],
                "max_tokens": self.config["max_output_tokens"],
                "stream": False,
            }
            if self.settings.get("system_prompt"):
                payload["messages"].insert(0, {"role": "system", "content": self.settings["system_prompt"]})
            url = self.base_url + "/chat/completions"
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
        started = time.monotonic()
        raw = request_json(url, payload, headers, self.timeout)
        elapsed = time.monotonic() - started
        if self.name == "gemini":
            candidate = (raw.get("candidates") or [{}])[0]
            text = "".join(
                p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
                if not p.get("thought")
            )
            usage = raw.get("usageMetadata", {})
            inp = usage.get("promptTokenCount", 0)
            visible = usage.get("candidatesTokenCount", 0)
            thinking = usage.get("thoughtsTokenCount", 0)
            cost = (inp * self.settings["input_usd_per_million"]
                    + (visible + thinking) * self.settings["output_usd_per_million"]) / 1e6
            finish = candidate.get("finishReason")
        else:
            choice = (raw.get("choices") or [{}])[0]
            text = choice.get("message", {}).get("content", "") or ""
            usage = raw.get("usage", {})
            inp = usage.get("prompt_tokens", 0)
            visible = usage.get("completion_tokens", 0)
            thinking = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            cost, finish = None, choice.get("finish_reason")
        return {
            "text": text, "raw_response": raw, "usage": usage,
            "input_tokens": inp, "output_tokens": visible,
            "thinking_tokens": thinking, "cost_usd": cost,
            "elapsed_seconds": elapsed, "finish_reason": finish,
            "returned_model": raw.get("modelVersion", raw.get("model", self.model)),
        }
