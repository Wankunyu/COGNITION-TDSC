import csv
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, Field


RUN_MANIFEST_SCHEMA_VERSION = "cognition.revision.run_manifest.v1"
ATTEMPT_RECORD_SCHEMA_VERSION = "cognition.revision.attempt.v1"
SUMMARY_ROW_SCHEMA_VERSION = "cognition.revision.summary_row.v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_DIRTY_CHECK_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "cognition/run_eval.py",
    "cognition/run_single_experiment.py",
    "cognition/revision_artifacts.py",
    "cognition/revision_preflight.py",
    "cognition/revision_secrets.py",
    "cognition/revision_provider_smoke.py",
    "tests",
)


class PromptConfig(BaseModel):
    prompt_mode: str
    prompts_file: str | None = None
    prompts_file_sha256: str | None = None
    few_shot_config: str | None = None
    few_shot_config_sha256: str | None = None
    prompt_prefix_sha256: str | None = None
    prompt_suffix_sha256: str | None = None


class RunManifest(BaseModel):
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str
    created_at: datetime
    code_revision: dict[str, Any]
    python_version: str
    dependency_versions: dict[str, str]
    dataset_summary: dict[str, Any]
    prompt_config: PromptConfig
    provider: str
    model: str
    seed: int | None = None
    retry_policy: dict[str, Any] = Field(default_factory=dict)
    cost_control: dict[str, Any] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
    ethics_scope: str = "offline authorized datasets only; no live service automation"


class AttemptRecord(BaseModel):
    schema_version: str = ATTEMPT_RECORD_SCHEMA_VERSION
    run_id: str
    attempt_id: str
    task_type: str
    puzzle_id: str
    attempt_index: int
    prompt_mode: str
    provider: str
    model: str
    parsed_answer: Any = None
    correct: bool
    error_category: str | None = None
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    timestamp: datetime


class SummaryRow(BaseModel):
    schema_version: str = SUMMARY_ROW_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    task_type: str
    n_attempts: int
    n_success: int
    pass_rate: float


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_code_revision() -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    dirty_output = _run_git(
        ["status", "--short", "--untracked-files=normal", "--", *_DIRTY_CHECK_PATHS]
    )
    return {
        "commit": commit,
        "dirty": bool(dirty_output),
        "dirty_scope": "source-test-and-doc-files-excluding-local-secret-config",
    }


def collect_dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def sha256_file(path: str | Path | None) -> str | None:
    if path is None or str(path) == "":
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def revision_run_dir(output_root: str | Path, run_id: str) -> Path:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or number and contain only letters, "
            "numbers, dot, underscore, or hyphen"
        )
    root = Path(output_root).resolve()
    run_dir = (root / run_id).resolve()
    if not run_dir.is_relative_to(root):
        raise ValueError("run_id resolves outside output_root")
    return run_dir


class RevisionArtifactWriter:
    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        *,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")

        self._run_dir = revision_run_dir(output_root, run_id)
        if self._run_dir.exists() and not overwrite and not resume:
            raise FileExistsError(f"Revision run directory already exists: {self._run_dir}")
        if self._run_dir.exists() and overwrite:
            shutil.rmtree(self._run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=resume)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def attempts_path(self) -> Path:
        return self.run_dir / "attempts.jsonl"

    @property
    def summary_csv_path(self) -> Path:
        return self.run_dir / "summary.csv"

    @property
    def summary_json_path(self) -> Path:
        return self.run_dir / "summary.json"

    def output_paths(self) -> dict[str, str]:
        return {
            "run_manifest": str(self.manifest_path),
            "attempts": str(self.attempts_path),
            "summary_csv": str(self.summary_csv_path),
            "summary_json": str(self.summary_json_path),
        }

    def write_manifest(self, manifest: RunManifest) -> Path:
        payload = manifest.model_dump(mode="json")
        payload["output_paths"] = payload.get("output_paths") or self.output_paths()
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return self.manifest_path

    def append_attempt(self, attempt: AttemptRecord) -> Path:
        existing_attempt_ids = {existing.attempt_id for existing in self.iter_attempts()}
        if attempt.attempt_id in existing_attempt_ids:
            raise ValueError(f"Attempt already exists in attempts.jsonl: {attempt.attempt_id}")
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(attempt.model_dump_json())
            handle.write("\n")
        return self.attempts_path

    def iter_attempts(self) -> Iterator[AttemptRecord]:
        if not self.attempts_path.exists():
            return
        with self.attempts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield AttemptRecord.model_validate_json(line)

    def write_summaries_from_attempts(self) -> tuple[Path, Path]:
        groups: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
            lambda: {"n_attempts": 0, "n_success": 0}
        )
        for attempt in self.iter_attempts():
            key = (attempt.run_id, attempt.provider, attempt.model, attempt.task_type)
            groups[key]["n_attempts"] += 1
            groups[key]["n_success"] += int(attempt.correct)

        rows = []
        for (run_id, provider, model, task_type), counts in sorted(groups.items()):
            n_attempts = counts["n_attempts"]
            n_success = counts["n_success"]
            rows.append(
                SummaryRow(
                    run_id=run_id,
                    provider=provider,
                    model=model,
                    task_type=task_type,
                    n_attempts=n_attempts,
                    n_success=n_success,
                    pass_rate=n_success / n_attempts if n_attempts else 0.0,
                )
            )

        fieldnames = list(SummaryRow.model_fields)
        with self.summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.model_dump(mode="json"))

        with self.summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": SUMMARY_ROW_SCHEMA_VERSION,
                    "rows": [row.model_dump(mode="json") for row in rows],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

        return self.summary_csv_path, self.summary_json_path
