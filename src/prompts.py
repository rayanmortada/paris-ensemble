"""Prompts live in prompts/*.md: a version line, then SYSTEM and USER sections with {placeholders}."""
from dataclasses import dataclass
from functools import lru_cache

from src.config import ROOT


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user: str

    def fill_system(self, **kw) -> str:
        return self.system.format(**kw)

    def fill_user(self, **kw) -> str:
        return self.user.format(**kw)


@lru_cache
def load(name: str) -> Prompt:
    text = (ROOT / "prompts" / f"{name}.md").read_text()
    head, rest = text.split("\n", 1)
    system, user = rest.split("\nUSER\n", 1)
    return Prompt(version=f"{name}:{head.split(':', 1)[1].strip()}",
                  system=system.removeprefix("SYSTEM\n").strip(), user=user.strip())
