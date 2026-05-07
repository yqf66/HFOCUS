from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_START_PREFIX = "<<<PROMPT:"
_PROMPT_START_SUFFIX = ">>>"
_PROMPT_END = "<<<END_PROMPT>>>"
_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().with_name("prompts.txt")


def _parse_prompt_blocks(text: str) -> dict[str, str]:
    prompts: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if current_key is None:
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(_PROMPT_START_PREFIX) and stripped.endswith(_PROMPT_START_SUFFIX):
                key = stripped[len(_PROMPT_START_PREFIX) : -len(_PROMPT_START_SUFFIX)].strip()
                if not key:
                    raise ValueError(f"Empty prompt key at line {line_no}.")
                if key in prompts:
                    raise ValueError(f"Duplicate prompt key '{key}' at line {line_no}.")
                current_key = key
                current_lines = []
                continue
            raise ValueError(
                f"Invalid prompt store format at line {line_no}: "
                "expected prompt marker, blank line, or comment."
            )

        if stripped == _PROMPT_END:
            prompts[current_key] = "\n".join(current_lines).strip("\n")
            current_key = None
            current_lines = []
            continue

        current_lines.append(line)

    if current_key is not None:
        raise ValueError(f"Prompt '{current_key}' is missing closing marker '{_PROMPT_END}'.")

    return prompts


@lru_cache(maxsize=None)
def _load_prompt_map_cached(resolved_path: str) -> dict[str, str]:
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt store file not found: {path}")
    return _parse_prompt_blocks(path.read_text(encoding="utf-8"))


def load_prompt_map(prompts_path: str | Path | None = None) -> dict[str, str]:
    path = Path(prompts_path).expanduser() if prompts_path else _DEFAULT_PROMPTS_PATH
    return _load_prompt_map_cached(str(path.resolve()))


def get_prompt(key: str, prompts_path: str | Path | None = None) -> str:
    prompts = load_prompt_map(prompts_path)
    if key not in prompts:
        raise KeyError(f"Prompt key '{key}' not found in {prompts_path or _DEFAULT_PROMPTS_PATH}.")
    return prompts[key]
