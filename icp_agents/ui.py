"""Live console: shows the agents' conversation as it happens."""
from __future__ import annotations

import textwrap

from .messages import AgentMessage

COLORS = {"Coordinator": "bold white", "Scout": "cyan", "Researcher": "green", "Analyst": "yellow",
          "Strategist": "magenta", "Reporter": "blue", "*": "bold white"}
ARROWS = {"request": "asks", "reply": "answers", "handoff": "hands off to", "inform": "tells",
          "challenge": "challenges"}


class ConsoleUI:
    def __init__(self, max_chars: int = 900, plain: bool = False):
        self.max_chars = max_chars
        self.console = None
        if not plain:
            try:
                from rich.console import Console
                self.console = Console(highlight=False)
            except ImportError:
                pass

    def __call__(self, m: AgentMessage) -> None:
        verb = ARROWS.get(m.performative, m.performative)
        if m.data.get("challenge"):
            verb = "challenges"
        to = "the team" if m.recipient == "*" else m.recipient
        text = m.text if len(m.text) <= self.max_chars else m.text[: self.max_chars] + " ..."
        ref = f" (re #{m.in_reply_to})" if m.in_reply_to else ""
        stamp = m.ts.astimezone().strftime("%H:%M:%S")
        if self.console:
            from rich.markup import escape
            s, r = COLORS.get(m.sender, "white"), COLORS.get(m.recipient, "white")
            self.console.print(f"[dim]{stamp} #{m.id}{ref}[/dim] [{s}]{m.sender}[/{s}] {verb} [{r}]{to}[/{r}]"
                               f" [dim]· {escape(m.subject)}[/dim]")
            self.console.print(textwrap.indent(textwrap.fill(escape(text), 110), "    "))
            self.console.print()
        else:
            print(f"{stamp} #{m.id}{ref} {m.sender} {verb} {to} · {m.subject}")
            print(textwrap.indent(textwrap.fill(text, 110), "    ") + "\n")
