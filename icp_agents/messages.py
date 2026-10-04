"""The message every agent sends to another agent.

A message is what one colleague would write to another: a natural-language
`text` that explains what is being handed over, asked, or answered, plus the
structured `data` the receiver needs to act on it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_ids = itertools.count(1)

# Performatives: what the sender intends the message to do.
INFORM = "inform"        # sharing a finding or status
HANDOFF = "handoff"      # passing work to the next agent
REQUEST = "request"      # a question the sender is waiting on
REPLY = "reply"          # the answer to a request
CHALLENGE = "challenge"  # disagreeing with another agent's conclusion
BROADCAST = "*"          # recipient meaning "the whole team"


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    performative: str
    subject: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    in_reply_to: int | None = None
    thread: str | None = None
    id: int = field(default_factory=lambda: next(_ids))
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id, "ts": self.ts.isoformat(timespec="milliseconds"), "sender": self.sender,
            "recipient": self.recipient, "performative": self.performative, "subject": self.subject,
            "text": self.text, "in_reply_to": self.in_reply_to, "thread": self.thread, "data": self.data,
        }
